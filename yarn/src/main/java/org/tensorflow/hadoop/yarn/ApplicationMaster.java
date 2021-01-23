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

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringReader;
import java.lang.reflect.UndeclaredThrowableException;
import java.net.InetSocketAddress;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.ByteBuffer;
import java.security.PrivilegedExceptionAction;
import java.util.*;
import java.util.Map.Entry;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.classification.InterfaceAudience;
import org.apache.hadoop.classification.InterfaceStability;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.PathNotFoundException;
import org.apache.hadoop.http.HttpServer2;
import org.apache.hadoop.io.DataOutputBuffer;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.net.NetUtils;
import org.apache.hadoop.registry.client.api.BindFlags;
import org.apache.hadoop.registry.client.api.RegistryConstants;
import org.apache.hadoop.registry.client.api.RegistryOperations;
import org.apache.hadoop.registry.client.api.RegistryOperationsFactory;
import org.apache.hadoop.registry.client.types.ServiceRecord;
import org.apache.hadoop.registry.client.types.yarn.PersistencePolicies;
import org.apache.hadoop.registry.client.types.yarn.YarnRegistryAttributes;
import org.apache.hadoop.security.Credentials;
import org.apache.hadoop.security.UserGroupInformation;
import org.apache.hadoop.security.token.Token;
import org.apache.hadoop.util.ExitUtil;
import org.apache.hadoop.util.Shell;
import org.apache.hadoop.yarn.api.ApplicationConstants;
import org.apache.hadoop.yarn.api.ApplicationConstants.Environment;
import org.apache.hadoop.yarn.api.ContainerManagementProtocol;
import org.apache.hadoop.yarn.api.protocolrecords.RegisterApplicationMasterResponse;
import org.apache.hadoop.yarn.api.records.ApplicationAttemptId;
import org.apache.hadoop.yarn.api.records.Container;
import org.apache.hadoop.yarn.api.records.ContainerExitStatus;
import org.apache.hadoop.yarn.api.records.ContainerId;
import org.apache.hadoop.yarn.api.records.ContainerLaunchContext;
import org.apache.hadoop.yarn.api.records.ContainerState;
import org.apache.hadoop.yarn.api.records.ContainerStatus;
import org.apache.hadoop.yarn.api.records.FinalApplicationStatus;
import org.apache.hadoop.yarn.api.records.LocalResource;
import org.apache.hadoop.yarn.api.records.NodeReport;
import org.apache.hadoop.yarn.api.records.Priority;
import org.apache.hadoop.yarn.api.records.Resource;
import org.apache.hadoop.yarn.api.records.timeline.TimelineEntity;
import org.apache.hadoop.yarn.api.records.timeline.TimelineEvent;
import org.apache.hadoop.yarn.api.records.timeline.TimelinePutResponse;
import org.apache.hadoop.yarn.client.api.AMRMClient.ContainerRequest;
import org.apache.hadoop.yarn.client.api.TimelineClient;
import org.apache.hadoop.yarn.client.api.async.AMRMClientAsync;
import org.apache.hadoop.yarn.client.api.async.NMClientAsync;
import org.apache.hadoop.yarn.client.api.async.impl.NMClientAsyncImpl;
import org.apache.hadoop.yarn.conf.YarnConfiguration;
import org.apache.hadoop.yarn.exceptions.YarnException;
import org.apache.hadoop.yarn.security.AMRMTokenIdentifier;
import org.apache.hadoop.yarn.util.ConverterUtils;
import org.apache.log4j.LogManager;

import com.google.common.annotations.VisibleForTesting;

@InterfaceAudience.Public
@InterfaceStability.Unstable
public class ApplicationMaster
{
    private static final Log LOG = LogFactory.getLog(ApplicationMaster.class);
    // used by taskStarter to pass back info to AM
    private static final String TASK_JOB_NAME = "task_job_name";
    private static final String TASK_JOB_INDEX = "task_job_index";
    private static final String TASK_PORT = "task_port";

    // Configuration
    private Configuration conf;

    // Handle to communicate with the Resource Manager
    private AMRMClientAsync amRMClient;

    // In both secure and non-secure modes, this points to the job-submitter.
    @VisibleForTesting
    UserGroupInformation appSubmitterUgi;

    // Handle to communicate with the Node Manager
    private NMClientAsync nmClientAsync;
    // Listen to process the response from the Node Manager
    private NMCallbackHandler containerListener;

    // Application Attempt Id ( combination of attemptId and fail count )
    @VisibleForTesting
    private ApplicationAttemptId appAttemptID;

    private String appIDString;

    // For status update for clients - yet to be implemented
    // Hostname of the container
    private String appMasterHostname = "";
    // Port on which the app master listens for status updates from clients
    private int appMasterRpcPort = -1;
    // Tracking url to which app master publishes info for clients to monitor
    private String appMasterTrackingUrl = "";

    // App Master configuration
    // No. of containers to run shell command on
    @VisibleForTesting
    private int numTotalContainers = 1;
    // Memory to request for the container on which the shell command will run
    private int containerMemory = 10;
    // VirtualCores to request for the container on which the shell command will
    // run
    private int containerVirtualCores = 1;
    // Priority of the request
    private int requestPriority;

    // Counter for completed containers ( complete denotes successful or failed
    // )
    private AtomicInteger numCompletedContainers = new AtomicInteger();
    // Allocated container count so that we know how many containers has the RM
    // allocated to us
    @VisibleForTesting
    private AtomicInteger numAllocatedContainers = new AtomicInteger();
    // Count of failed containers
    private AtomicInteger numFailedContainers = new AtomicInteger();
    // Count of containers already requested from the RM
    // Needed as once requested, we should not request for containers again.
    // Only request for more if the original requirement changes.
    @VisibleForTesting
    private AtomicInteger numRequestedContainers = new AtomicInteger();

    // Shell command to be executed
    private String taskCommand = "";
    // Args to be passed to the shell command
    private String taskArgs = "";
    // Env variables to be setup for the shell command
    private Map<String, String> shellEnv = new ConcurrentHashMap<>();

    // Location of shell script ( obtained from info set in env )
    // Shell script path in fs
    private String scriptPath = "";
    // Timestamp needed for creating a local resource
    private long shellScriptPathTimestamp = 0;
    // File length needed for local resource
    private long shellScriptPathLen = 0;

    private Map<String, LocalResource> localResources = new ConcurrentHashMap<>();

    // Timeline domain ID
    private String domainId = null;

    // Hardcoded path to custom log_properties
    private static final String log4jPath = "log4j.properties";

    private static final String shellCommandPath = "shellCommands";
    private static final String shellArgsPath = "shellArgs";

    private volatile boolean done;

    private ByteBuffer allTokens;

    // Launch threads
    private List<Thread> launchThreads = new ArrayList<>();

    // Timeline Client
    @VisibleForTesting
    TimelineClient timelineClient;

    private TFSession tfSession;

    private String appSubmitterUserName;

    private boolean tensorboardFlag = false;

    /**
     * @param args
     *            Command line args
     */
    public static void main(String[] args)
    {
        boolean result = false;
        try
        {
            ApplicationMaster appMaster = new ApplicationMaster();
            LOG.info("TF: Initializing ApplicationMaster");
            boolean doRun = appMaster.init(args);
            if (!doRun)
            {
                System.exit(0);
            }

            appMaster.run();
            result = appMaster.finish();
        } catch (Throwable t)
        {
            LOG.fatal("Error running ApplicationMaster", t);
            LogManager.shutdown();
            ExitUtil.terminate(1, t);
        }

        if (result)
        {
            LOG.info("Application Master completed successfully. exiting");
            System.exit(0);
        } else
        {
            LOG.info("Application Master failed. exiting");
            System.exit(2);
        }
    }

    /**
     * Dump out contents of $CWD and the environment to stdout for debugging
     */
    private void dumpOutDebugInfo()
    {

        LOG.info("Dump debug output");
        Map<String, String> envs = System.getenv();
        for (Map.Entry<String, String> env : envs.entrySet())
        {
            LOG.info("System env: key=" + env.getKey() + ", val="
                    + env.getValue());
            System.out.println("System env: key=" + env.getKey() + ", val="
                    + env.getValue());
        }

        BufferedReader buf = null;
        try
        {
            String lines = Shell.WINDOWS ? Shell
                    .execCommand("cmd", "/c", "dir") : Shell.execCommand("ls",
                    "-al");
            buf = new BufferedReader(new StringReader(lines));
            String line;
            while ((line = buf.readLine()) != null)
            {
                LOG.info("System CWD content: " + line);
                System.out.println("System CWD content: " + line);
            }
        } catch (IOException e)
        {
            e.printStackTrace();
        } finally
        {
            IOUtils.cleanup(LOG, buf);
        }
    }

    public ApplicationMaster()
    {
        // Set up the configuration
        conf = new YarnConfiguration();
    }

    /**
     * Parse command line options
     *
     * @param args
     *            Command line args
     * @return Whether init successful and run should be invoked
     * @throws ParseException
     * @throws IOException
     */
    public boolean init(String[] args) throws ParseException, IOException
    {
        Options opts = new Options();
        opts.addOption("app_attempt_id", true,
                "App Attempt ID. Not to be used unless for testing purposes");
        opts.addOption("shell_env", true,
                "Environment for shell script. Specified as env_key=env_val pairs");
        opts.addOption("input_path", true,
                "Input path of TensorFlow tasks");
        opts.addOption("output_path", true,
                "Output path of TensorFlow tasks");
        opts.addOption("container_memory", true,
                "Amount of memory in MB to be requested to run the shell command");
        opts.addOption("container_vcores", true,
                "Amount of virtual cores to be requested to run the shell command");
        opts.addOption("num_containers", true,
                "No. of containers on which the shell command needs to be executed");
        opts.addOption("priority", true, "Application Priority. Default 0");
        opts.addOption("debug", false, "Dump out debug information");
        opts.addOption("enable_tensorboard", false, "Start TensorBoard as part of job");
        opts.addOption("docker_image", true, "Docker image for running the tasks");
        opts.addOption("appname", true, "Application Name. Default:  " + TFConstants.DEFAULT_APPNAME);

        opts.addOption("help", false, "Print usage");
        CommandLine cliParser = new GnuParser().parse(opts, args);

        if (args.length == 0)
        {
            printUsage(opts);
            throw new IllegalArgumentException(
                    "No args specified for application master to initialize");
        }

        // Check whether customer log4j.properties file exists
        if (fileExist(log4jPath))
        {
            try
            {
                Log4jPropertyHelper.updateLog4jConfiguration(
                        ApplicationMaster.class, log4jPath);
            } catch (Exception e)
            {
                LOG.warn("Can not set up custom log4j properties. " + e);
            }
        }

        if (cliParser.hasOption("help"))
        {
            printUsage(opts);
            return false;
        }

        if (cliParser.hasOption("debug"))
        {
            dumpOutDebugInfo();
        }

        if (cliParser.hasOption("enable_tensorboard"))
        {
            tensorboardFlag = true;
        }

        Map<String, String> envs = System.getenv();

        if (!envs.containsKey(Environment.CONTAINER_ID.name()))
        {
            if (cliParser.hasOption("app_attempt_id"))
            {
                String appIdStr = cliParser
                        .getOptionValue("app_attempt_id", "");
                appAttemptID = ConverterUtils.toApplicationAttemptId(appIdStr);
            } else
            {
                throw new IllegalArgumentException(
                        "Application Attempt Id not set in the environment");
            }
        } else
        {
            ContainerId containerId = ConverterUtils.toContainerId(envs
                    .get(Environment.CONTAINER_ID.name()));
            appAttemptID = containerId.getApplicationAttemptId();
        }

        appIDString = appAttemptID.getApplicationId().toString();

        if (!envs.containsKey(ApplicationConstants.APP_SUBMIT_TIME_ENV))
        {
            throw new RuntimeException(ApplicationConstants.APP_SUBMIT_TIME_ENV
                    + " not set in the environment");
        }
        if (!envs.containsKey(Environment.NM_HOST.name()))
        {
            throw new RuntimeException(Environment.NM_HOST.name()
                    + " not set in the environment");
        }
        if (!envs.containsKey(Environment.NM_HTTP_PORT.name()))
        {
            throw new RuntimeException(Environment.NM_HTTP_PORT
                    + " not set in the environment");
        }
        if (!envs.containsKey(Environment.NM_PORT.name()))
        {
            throw new RuntimeException(Environment.NM_PORT.name()
                    + " not set in the environment");
        }

        LOG.info("Application master for app" + ", appId="
                + appAttemptID.getApplicationId().getId()
                + ", clustertimestamp="
                + appAttemptID.getApplicationId().getClusterTimestamp()
                + ", attemptId=" + appAttemptID.getAttemptId());

        if (!fileExist(shellCommandPath)
                && envs.get(TFConstants.SCRIPTLOCATION)
                        .isEmpty())
        {
            throw new IllegalArgumentException(
                    "No shell command or shell script specified to be executed by application master");
        }

        if (fileExist(shellCommandPath))
        {
            taskCommand = readContent(shellCommandPath);
        }

        if (fileExist(shellArgsPath))
        {
            taskArgs = readContent(shellArgsPath);
        }

        if (cliParser.hasOption("shell_env"))
        {
            String shellEnvs[] = cliParser.getOptionValues("shell_env");
            for (String env : shellEnvs)
            {
                env = env.trim();
                int index = env.indexOf('=');
                if (index == -1)
                {
                    shellEnv.put(env, "");
                    continue;
                }
                String key = env.substring(0, index);
                String val = "";
                if (index < (env.length() - 1))
                {
                    val = env.substring(index + 1);
                }
                shellEnv.put(key, val);
            }
        }

        if (envs.containsKey(TFConstants.SCRIPTLOCATION))
        {
            scriptPath = envs.get(TFConstants.SCRIPTLOCATION);

            if (envs.containsKey(TFConstants.SCRIPTTIMESTAMP))
            {
                shellScriptPathTimestamp = Long.parseLong(envs
                        .get(TFConstants.SCRIPTTIMESTAMP));
            }
            if (envs.containsKey(TFConstants.SHELLSCRIPTLEN))
            {
                shellScriptPathLen = Long.parseLong(envs
                        .get(TFConstants.SHELLSCRIPTLEN));
            }
            if (!scriptPath.isEmpty()
                    && (shellScriptPathTimestamp <= 0 || shellScriptPathLen <= 0))
            {
                LOG.error("Illegal values in env for shell script path"
                        + ", path=" + scriptPath + ", len="
                        + shellScriptPathLen + ", timestamp="
                        + shellScriptPathTimestamp);
                throw new IllegalArgumentException(
                        "Illegal values in env for shell script path");
            }
        }

        String input_path = cliParser.getOptionValue("input_path", "");
        String output_path = cliParser.getOptionValue("output_path", "");
        String dockerImage = cliParser.getOptionValue("docker_image", "");

        String appName = cliParser.getOptionValue("appname", TFConstants.DEFAULT_APPNAME);

        String registryQuorum = conf.get(RegistryConstants.KEY_REGISTRY_ZK_QUORUM);
        if (registryQuorum == null || registryQuorum.isEmpty())
        {
            throw new IllegalArgumentException(
                    "Undefined mandatoryconfiguration <"
                            + RegistryConstants.KEY_REGISTRY_ZK_QUORUM + ">");
        }

        if (envs.containsKey(TFConstants.TIMELINEDOMAIN))
        {
            domainId = envs.get(TFConstants.TIMELINEDOMAIN);
        }


        // default container vcores/memory/priority request
        containerMemory = Integer.parseInt(cliParser.getOptionValue(
                "container_memory", "10"));
        containerVirtualCores = Integer.parseInt(cliParser.getOptionValue(
                "container_vcores", "1"));
        requestPriority = Integer.parseInt(cliParser.getOptionValue("priority",
                "0"));

        String clusterReqString = cliParser.getOptionValue("num_containers", "");

        // create TensorFlow session object
        TFSession.TFSessionBuilder builder = new TFSession.TFSessionBuilder();
        builder.setClusterReqString(clusterReqString);
        builder.setAppName(appName);
        builder.setAppIDString(appIDString);
        builder.setTaskCmd(taskCommand);
        builder.setTaskArgs(taskArgs);
        builder.setEnableTensorBoard(tensorboardFlag);
        builder.setScriptPath(scriptPath);
        builder.setInputPath(input_path);
        builder.setOutputPath(output_path);
        builder.setDockerImage(dockerImage);
        builder.setRegistryQuorum(registryQuorum);
        TFContainerRequest defaultRequest = new TFContainerRequest(containerVirtualCores,
                containerMemory,
                requestPriority);
        builder.setDefaultContainerRequest(defaultRequest);
        tfSession = builder.build();

        // Add application wide environment variables
        tfSession.setAppGlobalEnv(shellEnv);

        ArrayList<TFContainerRequest> requests = tfSession.getRequiredContainers();
        numTotalContainers = requests.size();
        if (numTotalContainers == 0)
        {
            throw new IllegalArgumentException(
                    "Cannot run TensorFlow with no tasks");
        }

        return true;
    }


    /**
     * Helper function to print usage
     *
     * @param opts
     *            Parsed command line options
     */
    private void printUsage(Options opts)
    {
        new HelpFormatter().printHelp("ApplicationMaster", opts);
    }

    /**
     * Main run function for the application master
     *
     * @throws YarnException
     * @throws IOException
     * @throws URISyntaxException
     */
    @SuppressWarnings({ "unchecked" })
    public void run() throws YarnException, IOException, InterruptedException, URISyntaxException
    {
        LOG.info("Starting ApplicationMaster");

        // Note: Credentials, Token, UserGroupInformation, DataOutputBuffer
        // class
        // are marked as LimitedPrivate
        Credentials credentials = UserGroupInformation.getCurrentUser()
                .getCredentials();
        DataOutputBuffer dob = new DataOutputBuffer();
        credentials.writeTokenStorageToStream(dob);
        // Now remove the AM->RM token so that containers cannot access it.
        Iterator<Token<?>> iter = credentials.getAllTokens().iterator();
        LOG.info("Executing with tokens:");
        while (iter.hasNext())
        {
            Token<?> token = iter.next();
            LOG.info(token);
            if (token.getKind().equals(AMRMTokenIdentifier.KIND_NAME))
            {
                iter.remove();
            }
        }
        allTokens = ByteBuffer.wrap(dob.getData(), 0, dob.getLength());

        // Create appSubmitterUgi and add original tokens to it
        appSubmitterUserName = System
                .getenv(ApplicationConstants.Environment.USER.name());
        appSubmitterUgi = UserGroupInformation
                .createRemoteUser(appSubmitterUserName);
        appSubmitterUgi.addCredentials(credentials);

        AMRMClientAsync.CallbackHandler allocListener = new RMCallbackHandler();
        amRMClient = AMRMClientAsync.createAMRMClientAsync(1000, allocListener);
        amRMClient.init(conf);
        amRMClient.start();

        containerListener = createNMCallbackHandler();
        nmClientAsync = new NMClientAsyncImpl(containerListener);
        nmClientAsync.init(conf);
        nmClientAsync.start();

        startTimelineClient(conf);
        if (timelineClient != null)
        {
            publishApplicationAttemptEvent(timelineClient,
                    appAttemptID.toString(), TFEvent.TF_APP_ATTEMPT_START,
                    domainId, appSubmitterUgi);
        }

        // Setup local RPC Server to accept status requests directly from
        // clients

        String localHostname = NetUtils.getHostname();

        HttpServer2.Builder builder = new HttpServer2.Builder()
                .setName("test")
                .addEndpoint(URI.create("http://" + localHostname + ":0"))
                .setFindPort(true);

        HttpServer2 server = builder.build();
        server.setAttribute(ApplicationMaster.class.getName(), this);
        server.addServlet("status", "/", TFServlet.class);

        try
        {
            server.start();

            InetSocketAddress addr = server.getConnectorAddress(0);
            appMasterHostname = addr.getHostName();
            appMasterRpcPort = addr.getPort();
            appMasterTrackingUrl = "http://" + NetUtils.getHostPortString(addr);

            LOG.info(String.format("Server started: hostname=<%s>, port=<%d>, url=<%s>",
                    appMasterHostname, appMasterRpcPort, appMasterTrackingUrl));
        } catch (Exception e)
        {
            e.printStackTrace();
        }

        // Register self with ResourceManager
        // This will start heartbeating to the RM
        RegisterApplicationMasterResponse response = amRMClient
                .registerApplicationMaster(appMasterHostname, appMasterRpcPort,
                        appMasterTrackingUrl);
        // Dump out information about cluster capability as seen by the
        // resource manager
        int maxMem = response.getMaximumResourceCapability().getMemory();
        LOG.info("Max mem capability of resources in this cluster " + maxMem);

        int maxVCores = response.getMaximumResourceCapability()
                .getVirtualCores();
        LOG.info("Max vcores capability of resources in this cluster "
                + maxVCores);

        // A resource ask cannot exceed the max.
        if (containerMemory > maxMem)
        {
            LOG.info("Container memory specified above max threshold of cluster."
                    + " Using max value."
                    + ", specified="
                    + containerMemory
                    + ", max=" + maxMem);
            containerMemory = maxMem;
        }

        if (containerVirtualCores > maxVCores)
        {
            LOG.info("Container virtual cores specified above max threshold of cluster."
                    + " Using max value."
                    + ", specified="
                    + containerVirtualCores + ", max=" + maxVCores);
            containerVirtualCores = maxVCores;
        }

        // open HDFS and write out resources for the session
        FileSystem fs = FileSystem.get(conf);
        tfSession.createResources(fs, localResources);

        // Setup ask for containers from RM
        // Send request for containers to RM
        // Until we get our fully allocated quota, we keep on polling RM for
        // containers
        // Keep looping until all the containers are launched and shell script
        // executed on them ( regardless of success/failure).

        ArrayList<TFContainerRequest> requests = tfSession.getRequiredContainers();
        for (TFContainerRequest request : requests)
        {
            ContainerRequest containerAsk = setupContainerAskForRM(request);
            amRMClient.addContainerRequest(containerAsk);
        }

        numRequestedContainers.set(requests.size());
    }

    private boolean scanRegistryRecord()
    {
        boolean ready = false;

        RegistryOperations regOps = RegistryOperationsFactory.createInstance(conf);
        regOps.start();

        try
        {
            String parentPath = getRegistryBasePath() + "/components";

            LOG.info("TF: attempting to list parentPath=<" + parentPath + ">");

            List<String> components = regOps.list(parentPath);

            for (String componentName : components)
            {
                String path = String.format("%s/%s", parentPath, componentName);

                LOG.info("TF: attempting to get path=<" + path + ">");

                ServiceRecord record = regOps.resolve(path);

                LOG.info("TF: container_id=<" + componentName + "> record=<"
                        + record.toString() + ">");

                String jobName = record.get(TASK_JOB_NAME);
                int index = Integer.valueOf(record.get(TASK_JOB_INDEX));
                int port = Integer.valueOf(record.get(TASK_PORT));

                ready = tfSession.updateAllocatedPort(jobName, index, port);
                if (ready)
                    break;
            }
        }
        catch (PathNotFoundException e)
        {
            // path not created yet
            ready = false;
        }
        catch (Exception e)
        {
            LOG.error(e);
            e.printStackTrace();

        }
        finally
        {
            regOps.stop();
        }

        return ready;
    }

    private String getRegistryBasePath() {
        // Format:
        //     /users/{username}/{serviceclass}/{instancename}/components/{componentname}
        // NOTE:
        //     the real path will have "hadoop.registry.zk.root" prepended (default: /registry)
        return String.format("/users/%s/%s/%s", appSubmitterUserName,
                TFSession.REGISTRY_SERVICE_CLASS,
                this.appIDString);
    }

    private void setRegistryRecord(Map<String, String> varmap)
    {
        String path = getRegistryBasePath();

        ServiceRecord record = new ServiceRecord();
        record.set(YarnRegistryAttributes.YARN_ID, this.appIDString);
        record.set(YarnRegistryAttributes.YARN_PERSISTENCE, PersistencePolicies.APPLICATION);
        record.description = "YARN TensorFlow Application Master";

        for (Entry<String, String> entry : varmap.entrySet())
        {
            String jobName = entry.getKey();
            record.set(jobName, entry.getValue());
        }

        LOG.info(String.format("Setting registry record %s to %s", path, record));
        RegistryOperations regOps = RegistryOperationsFactory.createInstance(conf);
        regOps.start();

        try
        {
            regOps.bind(path, record, BindFlags.OVERWRITE);
        }
        catch (Exception e)
        {
            LOG.error(e);
        }
        finally
        {
            regOps.stop();
        }
    }

    @VisibleForTesting
    private void startTimelineClient(final Configuration conf) throws YarnException,
            IOException, InterruptedException
    {
        try
        {
            appSubmitterUgi.doAs(new PrivilegedExceptionAction<Void>()
            {
                @Override
                public Void run() throws Exception
                {
                    if (conf.getBoolean(
                            YarnConfiguration.TIMELINE_SERVICE_ENABLED,
                            YarnConfiguration.DEFAULT_TIMELINE_SERVICE_ENABLED))
                    {
                        // Creating the Timeline Client
                        timelineClient = TimelineClient.createTimelineClient();
                        timelineClient.init(conf);
                        timelineClient.start();
                    } else
                    {
                        timelineClient = null;
                        LOG.warn("Timeline service is not enabled");
                    }
                    return null;
                }
            });
        } catch (UndeclaredThrowableException e)
        {
            throw new YarnException(e.getCause());
        }
    }

    @VisibleForTesting
    private NMCallbackHandler createNMCallbackHandler()
    {
        return new NMCallbackHandler(this);
    }

    @VisibleForTesting
    private boolean finish()
    {
        boolean registrySet = false;

        // wait for completion.
        while (!done && (numCompletedContainers.get() != numTotalContainers))
        {
            try
            {
                if (!registrySet && numAllocatedContainers.get() >= numTotalContainers)
                {
                    boolean ready = scanRegistryRecord();
                    if (ready) {
                        LOG.info("TF: All tasks ready; signal tasks to start execution.");
                        this.setRegistryRecord(tfSession.getClusterSpec());
                        registrySet = true;
                    }
                    else
                    {
                        LOG.info("TF: Not all tasks are ready yet; check again soon.");
                    }
                    Thread.sleep(1000);
                }
                else
                {
                    // sleep
                    Thread.sleep(200);
                }

            } catch (InterruptedException ex)
            {
                LOG.error(ex);
            }
        }

        if (timelineClient != null)
        {
            publishApplicationAttemptEvent(timelineClient,
                    appAttemptID.toString(), TFEvent.TF_APP_ATTEMPT_END,
                    domainId, appSubmitterUgi);
        }

        // Join all launched threads
        // needed for when we time out
        // and we need to release containers
        for (Thread launchThread : launchThreads)
        {
            try
            {
                launchThread.join(10000);
            } catch (InterruptedException e)
            {
                LOG.info("Exception thrown in thread join: " + e.getMessage());
                e.printStackTrace();
            }
        }

        // When the application completes, it should stop all running containers
        LOG.info("Application completed. Stopping running containers");
        nmClientAsync.stop();

        // When the application completes, it should send a finish application
        // signal to the RM
        LOG.info("Application completed. Signalling finish to RM");

        boolean success = true;

        FinalApplicationStatus appStatus = tfSession.getFinalStatus();
        String appMessage = tfSession.getFinalMessage();
        if (appStatus != FinalApplicationStatus.SUCCEEDED)
        {
            LOG.info(appMessage);
            success = false;
        }

        try
        {
            amRMClient.unregisterApplicationMaster(appStatus, appMessage, null);
        } catch (YarnException | IOException ex)
        {
            LOG.error("Failed to unregister application", ex);
        }

        amRMClient.waitForServiceToStop(5000);
        amRMClient.stop();

        // Stop Timeline Client
        if (timelineClient != null)
        {
            timelineClient.stop();
        }

        return success;
    }

    @VisibleForTesting
    @InterfaceAudience.Private
    public enum TFEvent {
        TF_APP_ATTEMPT_START, TF_APP_ATTEMPT_END, TF_CONTAINER_START, TF_CONTAINER_END
    }

    @VisibleForTesting
    @InterfaceAudience.Private
    public enum TFEntity {
        TF_APP_ATTEMPT, TF_CONTAINER
    }

    @VisibleForTesting
    @InterfaceAudience.Private
    public enum TFInfo {
        TF_TASK_NAME, TF_EXIT_STATUS, TF_STATE
    }

    private class RMCallbackHandler implements AMRMClientAsync.CallbackHandler
    {
        @SuppressWarnings("unchecked")
        @Override
        public void onContainersCompleted(
                List<ContainerStatus> completedContainers)
        {
            LOG.info("Got response from RM for container ask, completedCnt="
                    + completedContainers.size());

            for (ContainerStatus containerStatus : completedContainers)
            {
                LOG.info(appAttemptID
                        + " got container status for containerID="
                        + containerStatus.getContainerId() + ", state="
                        + containerStatus.getState() + ", exitStatus="
                        + containerStatus.getExitStatus() + ", diagnostics="
                        + containerStatus.getDiagnostics());

                // non complete containers should not be here
                assert (containerStatus.getState() == ContainerState.COMPLETE);

                // increment counters for completed/failed containers
                int exitStatus = containerStatus.getExitStatus();

                tfSession.handleContainerTaskCompleted(containerStatus.getContainerId(), exitStatus);

                if (0 != exitStatus)
                {
                    if (ContainerExitStatus.ABORTED != exitStatus)
                    {
                        // failed, but counts as completed
                        numCompletedContainers.incrementAndGet();
                        numFailedContainers.incrementAndGet();
                    }
                    LOG.info("Container failed."
                            + ", containerId="
                            + containerStatus.getContainerId());
                }
                else
                {
                    // nothing to do
                    // container completed successfully
                    numCompletedContainers.incrementAndGet();
                    LOG.info("Container completed successfully."
                            + ", containerId="
                            + containerStatus.getContainerId());
                }

                if (timelineClient != null)
                {
                    publishContainerEndEvent(timelineClient, containerStatus,
                            domainId, appSubmitterUgi);
                }
            }

            // ask for more containers if any failed
            int askCount = numTotalContainers - numRequestedContainers.get();
            numRequestedContainers.addAndGet(askCount);

            if (askCount > 0)
            {
                for (int i = 0; i < askCount; ++i)
                {
                    ContainerRequest containerAsk = setupContainerAskForRM();
                    amRMClient.addContainerRequest(containerAsk);
                }
            }

            done = tfSession.isDone();

            LOG.info("TF: numTotalContainers=" + numTotalContainers
                    + ", numCompletedContainers=" + numCompletedContainers
                    + ", numFailedContainers=" + numFailedContainers
                    + ", askCount=" + askCount
                    + ", done=" + done);

        }

        @Override
        public void onContainersAllocated(List<Container> allocatedContainers)
        {
            LOG.info("Got response from RM for container ask, allocatedCnt="
                    + allocatedContainers.size());

            numAllocatedContainers.addAndGet(allocatedContainers.size());

            for (Container allocatedContainer : allocatedContainers)
            {
                LOG.info("Launching task on a new container."
                        + ", containerId=" + allocatedContainer.getId()
                        + ", containerNode="
                        + allocatedContainer.getNodeId().getHost() + ":"
                        + allocatedContainer.getNodeId().getPort()
                        + ", containerNodeURI="
                        + allocatedContainer.getNodeHttpAddress()
                        + ", containerResourceMemory"
                        + allocatedContainer.getResource().getMemory()
                        + ", containerResourceVirtualCores"
                        + allocatedContainer.getResource().getVirtualCores());
                // + ", containerToken"
                // +allocatedContainer.getContainerToken().getIdentifier().toString());

                LaunchContainerRunnable runnableLaunchContainer = new LaunchContainerRunnable(
                        allocatedContainer, containerListener);
                Thread launchThread = new Thread(runnableLaunchContainer);

                // launch and start the container on a separate thread to keep
                // the main thread unblocked
                // as all containers may not be allocated at one go.
                launchThreads.add(launchThread);
                launchThread.start();
            }
        }

        @Override
        public void onShutdownRequest()
        {
            done = true;
        }

        @Override
        public void onNodesUpdated(List<NodeReport> updatedNodes)
        {
        }

        @Override
        public float getProgress()
        {
            // set progress to deliver to RM on next heartbeat
            return (float) numCompletedContainers.get() / numTotalContainers;
        }

        @Override
        public void onError(Throwable e)
        {
            done = true;
            amRMClient.stop();
        }
    }

    @VisibleForTesting
    static class NMCallbackHandler implements NMClientAsync.CallbackHandler
    {

        private ConcurrentMap<ContainerId, Container> containers = new ConcurrentHashMap<>();
        private final ApplicationMaster applicationMaster;

        public NMCallbackHandler(ApplicationMaster applicationMaster)
        {
            this.applicationMaster = applicationMaster;
        }

        public void addContainer(ContainerId containerId, Container container)
        {
            containers.putIfAbsent(containerId, container);
        }

        @Override
        public void onContainerStopped(ContainerId containerId)
        {
            if (LOG.isDebugEnabled())
            {
                LOG.debug("Succeeded to stop Container " + containerId);
            }
            containers.remove(containerId);
        }

        @Override
        public void onContainerStatusReceived(ContainerId containerId,
                ContainerStatus containerStatus)
        {
            if (LOG.isDebugEnabled())
            {
                LOG.debug("Container Status: id=" + containerId + ", status="
                        + containerStatus);
            }
        }

        @Override
        public void onContainerStarted(ContainerId containerId,
                Map<String, ByteBuffer> allServiceResponse)
        {
            if (LOG.isDebugEnabled())
            {
                LOG.debug("Succeeded to start Container " + containerId);
            }
            Container container = containers.get(containerId);
            if (container != null)
            {
                applicationMaster.nmClientAsync.getContainerStatusAsync(
                        containerId, container.getNodeId());
            }
            if (applicationMaster.timelineClient != null)
            {
                ApplicationMaster.publishContainerStartEvent(
                        applicationMaster.timelineClient, container,
                        applicationMaster.domainId,
                        applicationMaster.appSubmitterUgi);
            }
        }

        @Override
        public void onStartContainerError(ContainerId containerId, Throwable t)
        {
            LOG.error("Failed to start Container " + containerId);
            containers.remove(containerId);
            applicationMaster.numCompletedContainers.incrementAndGet();
            applicationMaster.numFailedContainers.incrementAndGet();
        }

        @Override
        public void onGetContainerStatusError(ContainerId containerId,
                Throwable t)
        {
            LOG.error("Failed to query the status of Container " + containerId);
        }

        @Override
        public void onStopContainerError(ContainerId containerId, Throwable t)
        {
            LOG.error("Failed to stop Container " + containerId);
            containers.remove(containerId);
        }
    }

    /**
     * Thread to connect to the {@link ContainerManagementProtocol} and launch
     * the container that will execute the shell command.
     */
    private class LaunchContainerRunnable implements Runnable
    {

        // Allocated container
        Container container;

        NMCallbackHandler containerListener;

        /**
         * @param lcontainer
         *            Allocated container
         * @param containerListener
         *            Callback handler of the container
         */
        public LaunchContainerRunnable(Container lcontainer,
                NMCallbackHandler containerListener)
        {
            this.container = lcontainer;
            this.containerListener = containerListener;
        }

        @Override
        /*
          Connects to CM, sets up container launch context
          for shell command and eventually dispatches the container
          start request to the CM.
         */
        public void run()
        {
            LOG.info(String.format(
                    "TF: Adding container on host=<%s> to TensorFlowApp",
                    container.getNodeId().getHost()));

            // Set the local resources
            Map<String, LocalResource> myLocalResources = new ConcurrentHashMap<>(localResources);

            // make a copy of general environment variables, "addContainer"
            // will add more container specific ones to it.
            Map<String, String> myShellEnv = new ConcurrentHashMap<>(shellEnv);

            // add DTF_TASK_JOB_NAME and DTF_TASK_INDEX to myShellEnv
            boolean added = tfSession.addContainer(container, myShellEnv);
            if (!added)
            {
                LOG.info("TF: got extra container; releasing container id=<" + container.getId() + ">");
                numAllocatedContainers.decrementAndGet();
                amRMClient.releaseAssignedContainer(container.getId());
                return;
            }

            // Set the necessary command to execute on the allocated container
            Vector<CharSequence> vargs = new Vector<>(5);

            vargs.add(tfSession.getTaskStarterCommand());
            // Add log redirect params
            vargs.add("1>" + ApplicationConstants.LOG_DIR_EXPANSION_VAR
                    + "/stdout");
            vargs.add("2>" + ApplicationConstants.LOG_DIR_EXPANSION_VAR
                    + "/stderr");

            // Get final commmand
            StringBuilder command = new StringBuilder();
            for (CharSequence str : vargs)
            {
                command.append(str).append(" ");
            }

            List<String> commands = new ArrayList<>();
            commands.add(command.toString());

            LOG.info(String.format("TF: constructed wrapper command=<%s>", commands));

            // Set up ContainerLaunchContext, setting local resource,
            // environment, command and token for constructor.
            //
            // Note for tokens: Set up tokens for the container.
            // This is require for NodeManager and container to download files from DFS.
            ContainerLaunchContext ctx = ContainerLaunchContext.newInstance(
                    myLocalResources, myShellEnv, commands, null,
                    allTokens.duplicate(), null);
            containerListener.addContainer(container.getId(), container);
            nmClientAsync.startContainerAsync(container, ctx);
        }
    }

    /**
     * Setup the request that will be sent to the RM for the container ask.
     *
     * @return the setup ResourceRequest to be sent to RM
     */
    private ContainerRequest setupContainerAskForRM(int virtualCores, int memory, int priority)
    {
        // set the priority for the request
        Priority pri = Priority.newInstance(priority);

        // Set up resource type requirements
        Resource capability = Resource.newInstance(memory, virtualCores);

        // Currently no locality requirements.
        String[] nodes = null;
        String[] racks = null;

        ContainerRequest request = new ContainerRequest(capability, nodes, racks, pri);
        LOG.info("Requested container ask: " + request.toString());
        return request;
    }

    private ContainerRequest setupContainerAskForRM()
    {
        return setupContainerAskForRM(this.containerVirtualCores,
                    this.containerMemory, this.requestPriority);
    }

    private ContainerRequest setupContainerAskForRM(TFContainerRequest request)
    {
        return setupContainerAskForRM(request.getVirtualCores(),
                request.getMemory(), request.getPriority());
    }

    private boolean fileExist(String filePath)
    {
        return new File(filePath).exists();
    }

    private String readContent(String filePath) throws IOException
    {
        DataInputStream ds = null;
        try
        {
            ds = new DataInputStream(new FileInputStream(filePath));
            return ds.readUTF();
        } finally
        {
            org.apache.commons.io.IOUtils.closeQuietly(ds);
        }
    }

    private static void publishContainerStartEvent(
            final TimelineClient timelineClient, Container container,
            String domainId, UserGroupInformation ugi)
    {
        final TimelineEntity entity = new TimelineEntity();
        entity.setEntityId(container.getId().toString());
        entity.setEntityType(TFEntity.TF_CONTAINER.toString());
        entity.setDomainId(domainId);
        entity.addPrimaryFilter("user", ugi.getShortUserName());
        TimelineEvent event = new TimelineEvent();
        event.setTimestamp(System.currentTimeMillis());
        event.setEventType(TFEvent.TF_CONTAINER_START.toString());
        event.addEventInfo("Node", container.getNodeId().toString());
        event.addEventInfo("Resources", container.getResource().toString());
        entity.addEvent(event);

        try
        {
            ugi.doAs(new PrivilegedExceptionAction<TimelinePutResponse>()
            {
                @Override
                public TimelinePutResponse run() throws Exception
                {
                    return timelineClient.putEntities(entity);
                }
            });
        } catch (Exception e)
        {
            LOG.error("Container start event could not be published for "
                    + container.getId().toString(),
                    e instanceof UndeclaredThrowableException ? e.getCause()
                            : e);
        }
    }

    private void publishContainerEndEvent(
            final TimelineClient timelineClient, ContainerStatus container,
            String domainId, UserGroupInformation ugi)
    {
        final TimelineEntity entity = new TimelineEntity();
        entity.setEntityId(container.getContainerId().toString());
        entity.setEntityType(TFEntity.TF_CONTAINER.toString());
        entity.setDomainId(domainId);
        entity.addPrimaryFilter("user", ugi.getShortUserName());
        TimelineEvent event = new TimelineEvent();
        event.setTimestamp(System.currentTimeMillis());
        event.setEventType(TFEvent.TF_CONTAINER_END.toString());
        event.addEventInfo(TFInfo.TF_STATE.toString(), container.getState().name());
        event.addEventInfo(TFInfo.TF_EXIT_STATUS.toString(), container.getExitStatus());

        String taskName = tfSession.getJobAndIndex(container.getContainerId());
        event.addEventInfo(TFInfo.TF_TASK_NAME.toString(), taskName);

        entity.addEvent(event);
        try
        {
            timelineClient.putEntities(entity);
        } catch (YarnException | IOException e)
        {
            LOG.error("Container end event could not be published for "
                    + container.getContainerId().toString(), e);
        }
    }

    private static void publishApplicationAttemptEvent(
            final TimelineClient timelineClient, String appAttemptId,
            TFEvent appEvent, String domainId, UserGroupInformation ugi)
    {
        final TimelineEntity entity = new TimelineEntity();
        entity.setEntityId(appAttemptId);
        entity.setEntityType(TFEntity.TF_APP_ATTEMPT.toString());
        entity.setDomainId(domainId);
        entity.addPrimaryFilter("user", ugi.getShortUserName());
        TimelineEvent event = new TimelineEvent();
        event.setEventType(appEvent.toString());
        event.setTimestamp(System.currentTimeMillis());
        entity.addEvent(event);
        try
        {
            timelineClient.putEntities(entity);
        } catch (YarnException | IOException e)
        {
            LOG.error(
                    "App Attempt "
                            + (appEvent.equals(TFEvent.TF_APP_ATTEMPT_START) ? "start"
                                    : "end")
                            + " event could not be published for "
                            + appAttemptId, e);
        }
    }

    void printHtmlStatus(PrintWriter out)
    {
        out.println("<html><body>");
        out.println("<h1>YARN TensorFlow Application Status Page</h1>");

        tfSession.printHtmlStatusTable(out);

        out.println("</body></html>");
    }

}
