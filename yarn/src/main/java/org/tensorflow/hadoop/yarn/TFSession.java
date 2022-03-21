/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.hadoop.yarn;

/**
 * TensorFlow launcher for YARN
 *
 * Modified from YARN sample application: DistributedShell.
 */

import org.apache.commons.io.IOUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.permission.FsPermission;
import org.apache.hadoop.yarn.api.records.*;
import org.apache.hadoop.yarn.util.ConverterUtils;

import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

class TFSession {
    static final String REGISTRY_SERVICE_CLASS = "yarn-tensorflow";
    private static final Log LOG = LogFactory.getLog(TFSession.class);

    private static final String linux_bash_command = "bash";
    private static final String TaskProgramFile = "taskProgram";
    private static final String TaskScriptFile = "taskScript";
    private static final String TaskStarterPath = "taskStarter";
    private static final String TaskStarterModule = "wrapper";
    private static final String TensorBoardProgramFile = "tensorboardProgram";
    private static final String TaskStarterResource = "task-starter.zip";

    private static final String DTF_TENSORBOARD_JOBNAME = "TensorBoard";
    private static final String DTF_TASK_PROGRAM = "DTF_TASK_PROGRAM";
    private static final String DTF_TASK_SCRIPT = "DTF_TASK_SCRIPT";
    private static final String DTF_TASK_JOB_NAME = "DTF_TASK_JOB_NAME";
    private static final String DTF_TASK_INDEX = "DTF_TASK_INDEX";
    private static final String DTF_INPUT_PATH = "DTF_INPUT_PATH";
    private static final String DTF_OUTPUT_PATH = "DTF_OUTPUT_PATH";
    private static final String DTF_DOCKER_IMAGE = "DTF_DOCKER_IMAGE";
    private static final String DTF_APPLICATION_ID = "DTF_APPLICATION_ID";
    private static final String DTF_ZK_HOSTS = "DTF_ZK_HOSTS";
    private static final String DTF_SERVICE_CLASS = "DTF_SERVICE_CLASS";
    private static final String DTF_JOBNAME_HOSTS_FORMAT = "DTF_%s_HOSTS";

    private static final String DEF_PS_JOB_NAME = "ps" ;
    private static final String DEF_CHIEF_JOB_NAME = "worker";
    private static final int DEF_CHIEF_INDEX = -1;  // no chief defined by default

    public enum TaskType {
        TASK_TYPE_CHIEF, TASK_TYPE_PARAMETER_SERVER, TASK_TYPE_OTHERS
    }

    // Application Name
    private String appName = "";
    // Application ID
    private String appIDString;
    // Shell command to be executed
    private String taskCommand = "";
    // Args to be passed to the shell command
    private String taskArgs = "";
    // Location of shell script ( obtained from info set in env )
    // Shell script path in fs
    private String scriptPath = "";
    private String input_path = "";
    private String output_path = "";
    private String dockerImage = "";
    private String registryQuorum = "";
    private boolean enableTensorBoard = false;
    private TFContainerRequest defaultContainerRequest = new TFContainerRequest(1, 1024, 0);

    // chief = "worker[0]", ps = "ps[*]"
    // Env variables to be setup for the shell command
    private String psJobName = DEF_PS_JOB_NAME;
    private String chiefJobName = DEF_CHIEF_JOB_NAME;
    private int chiefIndex = DEF_CHIEF_INDEX;

    // map of job name => task
    private Map<String, Task[]> jobTasks = new ConcurrentHashMap<>();

    private boolean isDone = false;
    private FinalApplicationStatus finalStatus = FinalApplicationStatus.UNDEFINED;
    private String finalMessage = null;

    private TFSession(final TFSessionBuilder builder) {
        appName = builder.appName;
        appIDString = builder.appIDString;
        taskCommand = builder.taskCmd;
        taskArgs = builder.taskArgs;
        scriptPath = builder.scriptPath;
        input_path = builder.inputPath;
        output_path = builder.outputPath;
        dockerImage = builder.dockerImage;
        registryQuorum = builder.registryQuorum;
        enableTensorBoard = builder.enableTensorBoard;
        defaultContainerRequest = new TFContainerRequest(builder.defaultContainerRequest);

        psJobName = DEF_PS_JOB_NAME;
        chiefJobName = DEF_CHIEF_JOB_NAME;
        chiefIndex = DEF_CHIEF_INDEX;

        for (Map.Entry<String, Integer> entry : parseClusterRequirementString(builder.clusterReqString).entrySet()) {
            String job_name = entry.getKey();
            Integer nTasks = entry.getValue();

            // setup a task to hosts array, to keep track on which task needs
            // container
            Task[] tasks = new Task[nTasks];
            jobTasks.put(job_name, tasks);
        }
        if (enableTensorBoard) {
            // setup additional task for tensorboard
            jobTasks.put(DTF_TENSORBOARD_JOBNAME, new Task[1]);
        }
    }

    private LocalResource makeLocalResource(FileSystem fs, Path dst, LocalResourceType type) throws IOException {
        FileStatus scFileStatus = fs.getFileStatus(dst);
        return LocalResource.newInstance(
                ConverterUtils.getYarnUrlFromURI(dst.toUri()), type, LocalResourceVisibility.APPLICATION,
                scFileStatus.getLen(), scFileStatus.getModificationTime());
    }

    synchronized ArrayList<TFContainerRequest> getRequiredContainers() {
        ArrayList<TFContainerRequest> requests = new ArrayList<>();

        for (Map.Entry<String, Task[]> entry : jobTasks.entrySet()) {
            Task[] tasks = entry.getValue();
            for (Task task : tasks) {
                if (task == null) {
                    // current, all tasks are assume to have the same container request
                    // potential enhancement for a job and/or task specific request
                    TFContainerRequest request = new TFContainerRequest(defaultContainerRequest);
                    requests.add(request);
                }
            }
        }

        return requests;
    }

    private synchronized boolean checkAllReady() {
        boolean ready = true;

        for (Map.Entry<String, Task[]> entry : jobTasks.entrySet()) {
            Task[] tasks = entry.getValue();
            for (Task task : tasks) {
                if (task == null || !task.isReady()) {
                    ready = false;
                    break;
                }
            }
        }

        if (ready) {
            notifyAll();
        }

        return ready;
    }

    private Task getTaskByContainerId(ContainerId id) {
        for (Map.Entry<String, Task[]> entry : jobTasks.entrySet()) {
            Task[] tasks = entry.getValue();
            for (Task task : tasks) {
                ContainerId containerId = task.getContainerId();
                if (containerId != null && containerId.equals(id)) {
                    return task;
                }
            }
        }

        return null;
    }

    String getTaskStarterCommand() {
        // get executable command
        return String.format("( PYTHONPATH=%s:$PYTHONPATH python -m %s.__main__ --debug )", TaskStarterPath, TaskStarterModule);
    }

    String getJobAndIndex(ContainerId containerId) {
        for (Map.Entry<String, Task[]> entry : jobTasks.entrySet()) {
            Task[] tasks = entry.getValue();
            for (Task task : tasks) {
                if (task != null && task.getContainerId().equals(containerId)) {
                    return String.format("%s[%d]", task.getJobName(), task.getIndex());
                }
            }
        }

        return "unknown[#]";
    }

    Map<String, String> getClusterSpec() {
        Map<String, String> map = new HashMap<>();

        for (Map.Entry<String, Task[]> entry : jobTasks.entrySet()) {
            String jobName = entry.getKey();
            Task[] tasks = entry.getValue();

            boolean first = true;
            StringBuilder builder = new StringBuilder();
            for (Task task : tasks) {
                if (task == null) {
                    continue;
                }

                String hostPort = task.getHostPort();
                if (!first)
                    builder.append(",");
                first = false;
                builder.append(hostPort);
            }
            String jobNameEnv = convertJobNameEnv(jobName);
            map.put(jobNameEnv, builder.toString());
        }

        return map;
    }

    synchronized boolean updateAllocatedPort(String jobName, int index, int port) {
        Task[] tasks = jobTasks.get(jobName);

        if (tasks != null && index >= 0 && index < tasks.length && tasks[index] != null)
            tasks[index].setPort(port);
        return checkAllReady();
    }

    void setAppGlobalEnv(Map<String, String> shellEnv) {
        shellEnv.put(DTF_SERVICE_CLASS, REGISTRY_SERVICE_CLASS);
        shellEnv.put(DTF_APPLICATION_ID, appIDString);
        shellEnv.put(DTF_ZK_HOSTS, registryQuorum);

        if (!input_path.isEmpty()) {
            shellEnv.put(DTF_INPUT_PATH, input_path);
        }
        if (!output_path.isEmpty()) {
            shellEnv.put(DTF_OUTPUT_PATH, output_path);
        }
        if (!dockerImage.isEmpty()) {
            shellEnv.put(DTF_DOCKER_IMAGE, dockerImage);
        }
        if (!scriptPath.isEmpty()) {
            shellEnv.put(DTF_TASK_SCRIPT, TaskScriptFile);
        }
        shellEnv.put(DTF_TASK_PROGRAM, TaskProgramFile);
    }

    private String getFsBaseDir() {
        return appName + "/" + appIDString;
    }

    private void writeBytesToFs(FileSystem fs, Path dst, String content) throws IOException {
        FSDataOutputStream ostream = null;
        try {
            ostream = FileSystem.create(fs, dst, new FsPermission((short) 0710));
            ostream.writeBytes(content);
        } finally {
            IOUtils.closeQuietly(ostream);
        }
    }

    private void writeBytesToFs(FileSystem fs, Path dst, InputStream content) throws IOException {
        FSDataOutputStream ostream = null;
        try {
            ostream = FileSystem.create(fs, dst, new FsPermission((short) 0710));
            IOUtils.copy(content, ostream);
        } finally {
            IOUtils.closeQuietly(ostream);
        }
    }

    private Path writeTaskStarterToFs(FileSystem fs) throws IOException {
        ClassLoader clsLoader = getClass().getClassLoader();
        InputStream inStream = clsLoader.getResourceAsStream(TaskStarterResource);

        String baseDir = getFsBaseDir() + "/" + TaskStarterResource;
        Path dst = new Path(fs.getHomeDirectory(), baseDir);

        writeBytesToFs(fs, dst, inStream);

        return dst;
    }

    private Path writeTensorBoardProgramToFs(FileSystem fs) throws IOException {
        String taskProgramText;
        String hostsEnvVar = convertJobNameEnv(DTF_TENSORBOARD_JOBNAME);
        taskProgramText = "PORT=$(echo ${" + hostsEnvVar + "} | sed 's#.*:##'); tensorboard --port ${PORT} --logdir ${DTF_OUTPUT_PATH}";

        String baseDir = getFsBaseDir() + "/" + TensorBoardProgramFile;
        Path dst = new Path(fs.getHomeDirectory(), baseDir);

        writeBytesToFs(fs, dst, taskProgramText);

        return dst;
    }

    void createResources(FileSystem fs, Map<String, LocalResource> localResources) throws IOException {
        Path dst;
        LocalResource scRsrc;

        // add taskStarter to localResources
        dst = writeTaskStarterToFs(fs);
        scRsrc = makeLocalResource(fs, dst, LocalResourceType.ARCHIVE);
        localResources.put(TaskStarterPath, scRsrc);

        // add script to localResources
        if (!scriptPath.isEmpty()) {
            dst = new Path(scriptPath);
            scRsrc = makeLocalResource(fs, dst, LocalResourceType.FILE);
            localResources.put(TaskScriptFile, scRsrc);
        }

        dst = writeTaskProgramToFs(fs);
        scRsrc = makeLocalResource(fs, dst, LocalResourceType.FILE);
        localResources.put(TaskProgramFile, scRsrc);

        if (enableTensorBoard) {
            dst = writeTensorBoardProgramToFs(fs);
            scRsrc = makeLocalResource(fs, dst, LocalResourceType.FILE);
            localResources.put(TensorBoardProgramFile, scRsrc);
        }
    }

    private Path writeTaskProgramToFs(FileSystem fs) throws IOException {
        String taskProgramText;
        if (!taskCommand.isEmpty()) {
            // command args
            taskProgramText = taskCommand + " " + taskArgs;
        } else if (!scriptPath.isEmpty()) {
            // bash /c ExecScript.sh args...
            taskProgramText = linux_bash_command + " " + TaskScriptFile + " " + taskArgs;
        } else {
            // error, both script and shellCommand are empty
            throw new IllegalArgumentException("No task command nor task script provided");
        }

        String baseDir = getFsBaseDir() + "/" + TaskProgramFile;
        Path dst = new Path(fs.getHomeDirectory(), baseDir);

        writeBytesToFs(fs, dst, taskProgramText);

        return dst;
    }

    private static Map<String, Integer> parseClusterRequirementString(String clusterReqString) {
        Map<String, Integer> map = new ConcurrentHashMap<>();

        String[] jobs = clusterReqString.split(",");
        for (String jobName : jobs) {
            String[] job = jobName.split(":");
            if (job.length != 2) {
                throw new IllegalArgumentException(
                        "Invalid cluster requirement string <" + clusterReqString + ">");
            }
            map.put(job[0], Integer.valueOf(job[1]));
        }
        return map;
    }

    boolean isDone() {
        if (isDone)
            return true;

        int failedCnt = 0;

        // check
        for (Map.Entry<String, Task[]> entry : jobTasks.entrySet()) {
            String jobName = entry.getKey();
            Task[] tasks = entry.getValue();

            if (jobName.equals(psJobName) || jobName.equals(DTF_TENSORBOARD_JOBNAME)) {
                // ignore PS and TB job
                continue;
            }

            for (Task task : tasks) {
                if (task == null) {
                    LOG.info("TF: task is not started yet, isDone=false");
                    return false;
                }
                boolean isCompleted = task.isCompleted();
                if (!isCompleted) {
                    LOG.info("TF: task=" + task + ", is not completed yet, isDone=false");
                    return false;
                }

                int exitStatus = task.getExitStatus();
                if (exitStatus != 0) {
                    failedCnt++;
                }
            }
        }

        isDone = true;
        if (failedCnt > 0) {
            setFinalStatus(FinalApplicationStatus.FAILED,
                    "At least one job task exited with non-zero status, failedCnt="
                            + failedCnt);
        } else {
            setFinalStatus(FinalApplicationStatus.SUCCEEDED, null);
        }
        return isDone;
    }

    private TaskType getTaskType(Task task) {
        TaskType type;

        int index = task.getIndex();
        String jobName = task.getJobName();

        if (index == chiefIndex && jobName.equals(chiefJobName))
            type = TaskType.TASK_TYPE_CHIEF;
        else if (jobName.equals(psJobName))
            type = TaskType.TASK_TYPE_PARAMETER_SERVER;
        else
            type = TaskType.TASK_TYPE_OTHERS;

        return type;
    }

    boolean handleContainerTaskCompleted(ContainerId conainterId,
                                         int exitStatus) {
        Task task = getTaskByContainerId(conainterId);
        if (task == null) {
            return false;
        }

        TaskType taskType = getTaskType(task);

        LOG.info("TF: handleContainerTaskCompleted(): container=" + task.containerId + ", exitStatus=" + exitStatus
                + ", taskType=" + taskType);

        task.setExitStatus(exitStatus);

        switch (taskType) {
            case TASK_TYPE_CHIEF:
            case TASK_TYPE_OTHERS:
                if (exitStatus != 0) {
                    isDone = true;
                    setFinalStatus(FinalApplicationStatus.FAILED,
                            "Failed: a worker task exited with exitStatus=" + exitStatus + ", exiting application");
                }
                break;
            case TASK_TYPE_PARAMETER_SERVER:
                break;
            default:
                // not a TF task
                break;
        } // END of switch(taskType)

        LOG.info("TF: container=" + task.containerId + ", isDone=" + isDone
                + ", finalStatus=" + finalStatus
                + ", finalMessage=" + (finalMessage != null ? finalMessage : "null"));

        return isDone;
    }


    private void setFinalStatus(FinalApplicationStatus status,
                                String message) {
        finalStatus = status;
        finalMessage = message;
    }

    String getFinalMessage() {
        return finalMessage;
    }

    FinalApplicationStatus getFinalStatus() {
        return finalStatus;
    }

    private String convertJobNameEnv(String jobName) {
        return String.format(DTF_JOBNAME_HOSTS_FORMAT, jobName.toUpperCase());
    }

    synchronized boolean addContainer(Container container,
                                      Map<String, String> myShellEnv) {
        String host = container.getNodeId().getHost();

        // find a job+task to assign the container host
        for (Map.Entry<String, Task[]> entry : jobTasks.entrySet()) {
            String jobName = entry.getKey();
            Task[] tasks = entry.getValue();

            for (int i = 0; i < tasks.length; i++) {
                if (tasks[i] == null) {
                    tasks[i] = new Task(jobName, host, i, container.getId());
                    myShellEnv.put(DTF_TASK_JOB_NAME, jobName);
                    myShellEnv.put(DTF_TASK_INDEX, String.valueOf(i));

                    if (jobName.equals(DTF_TENSORBOARD_JOBNAME)) {
                        // overwrite task program with TensorBoard executable
                        myShellEnv.put(DTF_TASK_PROGRAM, TensorBoardProgramFile);
                    }

                    return true;
                }
            }
        }

        return false;
    }

    void printHtmlStatusTable(PrintWriter out) {

        if (jobTasks == null) {
            out.println("Information not available yet, try again later");
            return;
        }

        out.println("<table border=1>");
        out.println("<tr><th>Job Name</th><th>Index</th><th>HostPort</th><th>ContainerID</th><th>exitStatus</th></tr>");
        for (Map.Entry<String, Task[]> entry : jobTasks.entrySet()) {
            String jobName = entry.getKey();
            Task[] tasks = entry.getValue();

            out.println("<tr>");
            out.println("<td rowspan=" + tasks.length + ">" + jobName + "</td>");
            for (int i = 0; i < tasks.length; i++) {
                Task task = tasks[i];

                if (i != 0)
                    out.println("<tr>");
                out.println("<td>" + task.getIndex() + "</td>");
                if (jobName.equals(DTF_TENSORBOARD_JOBNAME) && task.isReady()) {
                    String hostPort = task.getHostPort();
                    out.println(String.format("<td><a href=\"http://%s\">%s</a>",
                            hostPort, hostPort));
                } else {
                    out.println("<td>" + task.getHostPort() + "</td>");
                }
                out.println("<td>" + task.getContainerId() + "</td>");
                out.println("<td>" + task.getExitStatus() + "</td>");
                out.println("</tr>");
            }
            out.println("</tr>");
        }
        out.println("</table>");

    }

    public static class Task {
        static private final String FORMAT_HOST_PORT = "%s:%d";

        private String jobName = "";
        private int taskIndex = -1;
        private String host = "";
        private int port = -1;
        ContainerId containerId = null;
        boolean completed = false;
        int exitStatus = -1;

        Task(String name, String host, int index, ContainerId id) {
            this.jobName = name;
            this.host = host;
            this.taskIndex = index;
            this.containerId = id;
        }

        public String toString() {
            return String.format("Task:%s[%d]/%s:%d/%s/%d", jobName, taskIndex, host, port, containerId, exitStatus);
        }

        String getJobName() {
            return this.jobName;
        }

        int getIndex() {
            return this.taskIndex;
        }

        String getHostPort() {
            return String.format(FORMAT_HOST_PORT, host, port < 0 ? 0 : port);
        }

        void setPort(int port) {
            this.port = port;
        }

        ContainerId getContainerId() {
            return this.containerId;
        }

        void setExitStatus(int status) {
            this.completed = true;
            this.exitStatus = status;
        }

        boolean isCompleted() {
            return this.completed;
        }

        int getExitStatus() {
            return this.exitStatus;
        }

        boolean isReady() {
            return (!host.isEmpty() && port > 0);
        }
    }

    static class TFSessionBuilder {
        private String clusterReqString;
        private String appIDString;
        private String taskCmd;
        private String taskArgs;
        private boolean enableTensorBoard;
        private String scriptPath;
        private String inputPath;
        private String outputPath;
        private String dockerImage;
        private String registryQuorum;
        private String appName;
        private TFContainerRequest defaultContainerRequest;

        TFSession build() {
            return new TFSession(this);
        }

        TFSessionBuilder setClusterReqString(String clusterReqString) {
            this.clusterReqString = clusterReqString;
            return this;
        }

        TFSessionBuilder setAppIDString(String appIDString) {
            this.appIDString = appIDString;
            return this;
        }

        TFSessionBuilder setTaskCmd(String taskCmd) {
            this.taskCmd = taskCmd;
            return this;
        }

        TFSessionBuilder setTaskArgs(String taskArgs) {
            this.taskArgs = taskArgs;
            return this;
        }

        TFSessionBuilder setEnableTensorBoard(boolean enableTensorBoard) {
            this.enableTensorBoard = enableTensorBoard;
            return this;
        }

        TFSessionBuilder setScriptPath(String scriptPath) {
            this.scriptPath = scriptPath;
            return this;
        }

        TFSessionBuilder setInputPath(String inputPath) {
            this.inputPath = inputPath;
            return this;
        }

        TFSessionBuilder setOutputPath(String outputPath) {
            this.outputPath = outputPath;
            return this;
        }

        TFSessionBuilder setDockerImage(String dockerImage) {
            this.dockerImage = dockerImage;
            return this;
        }

        TFSessionBuilder setRegistryQuorum(String registryQuorum) {
            this.registryQuorum = registryQuorum;
            return this;
        }

        TFSessionBuilder setAppName(String appName) {
            this.appName = appName;
            return this;
        }

        TFSessionBuilder setDefaultContainerRequest(TFContainerRequest defaultRequest) {
            this.defaultContainerRequest = defaultRequest;
            return this;
        }

    }
}
