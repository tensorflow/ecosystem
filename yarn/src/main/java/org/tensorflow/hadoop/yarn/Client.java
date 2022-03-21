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

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Vector;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.io.IOUtils;
import org.apache.commons.lang.StringUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.classification.InterfaceAudience;
import org.apache.hadoop.classification.InterfaceStability;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.permission.FsPermission;
import org.apache.hadoop.io.DataOutputBuffer;
import org.apache.hadoop.security.Credentials;
import org.apache.hadoop.security.UserGroupInformation;
import org.apache.hadoop.security.token.Token;
import org.apache.hadoop.yarn.api.ApplicationConstants;
import org.apache.hadoop.yarn.api.ApplicationConstants.Environment;
import org.apache.hadoop.yarn.api.protocolrecords.GetNewApplicationResponse;
import org.apache.hadoop.yarn.api.records.ApplicationId;
import org.apache.hadoop.yarn.api.records.ApplicationSubmissionContext;
import org.apache.hadoop.yarn.api.records.ContainerLaunchContext;
import org.apache.hadoop.yarn.api.records.LocalResource;
import org.apache.hadoop.yarn.api.records.LocalResourceType;
import org.apache.hadoop.yarn.api.records.LocalResourceVisibility;
import org.apache.hadoop.yarn.api.records.NodeReport;
import org.apache.hadoop.yarn.api.records.NodeState;
import org.apache.hadoop.yarn.api.records.Priority;
import org.apache.hadoop.yarn.api.records.QueueACL;
import org.apache.hadoop.yarn.api.records.QueueInfo;
import org.apache.hadoop.yarn.api.records.QueueUserACLInfo;
import org.apache.hadoop.yarn.api.records.Resource;
import org.apache.hadoop.yarn.api.records.YarnClusterMetrics;
import org.apache.hadoop.yarn.api.records.timeline.TimelineDomain;
import org.apache.hadoop.yarn.client.api.TimelineClient;
import org.apache.hadoop.yarn.client.api.YarnClient;
import org.apache.hadoop.yarn.client.api.YarnClientApplication;
import org.apache.hadoop.yarn.conf.YarnConfiguration;
import org.apache.hadoop.yarn.exceptions.YarnException;
import org.apache.hadoop.yarn.util.ConverterUtils;
import org.apache.hadoop.yarn.util.timeline.TimelineUtils;

@InterfaceAudience.Public
@InterfaceStability.Unstable
public class Client {

  private static final Log LOG = LogFactory.getLog(Client.class);
  
  // Configuration
  private Configuration conf;
  private YarnClient yarnClient;
  // Application master specific info to register a new Application with RM/ASM
  private String appName = "";
  // App master priority
  private int amPriority = 0;
  // Queue for App master
  private String amQueue = "";
  // Amt. of memory resource to request for to run the App Master
  private int amMemory = 10; 
  // Amt. of virtual core resource to request for to run the App Master
  private int amVCores = 1;

  // Application master jar file
  private String appMasterJar = ""; 
  // Main class to invoke application master
  private final String appMasterMainClass;

  // Shell command to be executed 
  private String taskCmd = ""; 
  // Location of shell script 
  private String taskScript = ""; 
  // Args to be passed to the shell command
  private String[] taskArgs = new String[] {};
  // Input path
  private String inputPath =  "";
  // Output path
  private String outputPath = "";
  // Docker engine to run the task command
  private String dockerImage = "";
  // Env variables to be setup for the shell command
  private Map<String, String> shellEnv = new HashMap<>();
  // Shell Command Container priority 
  private int shellCmdPriority = 0;

  // Amt of memory to request for container in which shell script will be executed
  private int containerMemory = 10; 
  // Amt. of virtual cores to request for container in which shell script will be executed
  private int containerVirtualCores = 1;
  // No. of containers in which the shell script needs to be executed
  private String numContainers = "";
  private String nodeLabelExpression = null;

  // log4j.properties file 
  // if available, add to local resources and set into classpath 
  private String log4jPropFile = "";	

  // flag to indicate whether to keep containers across application attempts.
  private boolean keepContainers = false;

  private long attemptFailuresValidityInterval = -1;

  // Debug flag
  private boolean debugFlag = false;

  // Timeline domain ID
  private String domainId = null;

  // Flag to indicate whether to create the domain of the given ID
  private boolean toCreateDomain = false;

  // Timeline domain reader access control
  private String viewACLs = null;

  // Timeline domain writer access control
  private String modifyACLs = null;

  // Command line options
  private Options opts;

  private boolean tensorboardFlag = false;

  private static final String shellCommandPath = "shellCommands";
  private static final String shellArgsPath = "shellArgs";
  private static final String appMasterJarPath = "AppMaster.jar";
  // Hardcoded path to custom log_properties
  private static final String log4jPath = "log4j.properties";

  public static final String SCRIPT_PATH = "ExecScript";

  /**
   * @param args Command line arguments 
   */
  public static void main(String[] args) {
    boolean result = false;
    try {
      Client client = new Client();
      LOG.info("Initializing Client");
      try {
        boolean doRun = client.init(args);
        if (!doRun) {
          System.exit(0);
        }
      } catch (IllegalArgumentException e) {
        System.err.println(e.getLocalizedMessage());
        client.printUsage();
        System.exit(-1);
      }
      result = client.run();
    } catch (Throwable t) {
      LOG.fatal("Error running Client", t);
      System.exit(1);
    }
    if (result) {
      LOG.info("Application submitted successfully");
      System.exit(0);			
    } 
    LOG.error("Application failed to submitted successfully");
    System.exit(2);
  }

  /**
   */
  public Client(Configuration conf) throws Exception  {
    this(
      ApplicationMaster.class.getName(),
      conf);
  }

  Client(String appMasterMainClass, Configuration conf) {
    this.conf = conf;
    this.appMasterMainClass = appMasterMainClass;
    yarnClient = YarnClient.createYarnClient();
    yarnClient.init(conf);
    opts = new Options();
    opts.addOption("appname", true, "Application Name. Default: " + TFConstants.DEFAULT_APPNAME);
    opts.addOption("priority", true, "Application Priority. Default 0");
    opts.addOption("queue", true, "RM Queue in which this application is to be submitted");
    opts.addOption("master_memory", true, "Amount of memory in MB to be requested to run the application master");
    opts.addOption("master_vcores", true, "Amount of virtual cores to be requested to run the application master");
    opts.addOption("jar", true, "Jar file containing the application master");
    opts.addOption("task_script", true, "Location of the task script, it will be copied to task execution environment.");
    opts.addOption("task_cmd", true, "Task execute command. At least one of 'task_script' or 'task_cmd' must be defined.");
    opts.addOption("task_args", true, "Command line args for the task program." +
        "Multiple args can be separated by empty space.");
    opts.getOption("task_args").setArgs(Option.UNLIMITED_VALUES);
    opts.addOption("input_path", true, "Input path for task program");
    opts.addOption("output_path", true, "Output path for task program");
    opts.addOption("docker_image", true, "Docker image for running the tasks");
    opts.addOption("shell_env", true, "Environment for task program. Specified as env_key=env_val pairs");
    opts.addOption("container_memory", true, "Amount of memory in MB to be requested to run the task program");
    opts.addOption("container_vcores", true, "Amount of virtual cores to be requested to run the task program");
    opts.addOption("num_containers", true, "Format <job_name>:<num_task>, separated by comma.  (ie: ps:2,worker:4)");
    opts.addOption("log_properties", true, "log4j.properties file");
    opts.addOption("enable_tensorboard", false, "Start TensorBoard as part of job");
    opts.addOption("debug", false, "Dump out debug information");    
    opts.addOption("domain", true, "ID of the timeline domain where the "
        + "timeline entities will be put");
    opts.addOption("view_acls", true, "Users and groups that allowed to "
        + "view the timeline entities in the given domain");
    opts.addOption("modify_acls", true, "Users and groups that allowed to "
        + "modify the timeline entities in the given domain");
    opts.addOption("create", false, "Flag to indicate whether to create the "
        + "domain specified with -domain.");
    opts.addOption("help", false, "Print usage");

    opts.addOption("node_label_expression", true,
        "Node label expression to determine the nodes"
            + " where all the containers of this application"
            + " will be allocated, \"\" means containers"
            + " can be allocated anywhere, if you don't specify the option,"
            + " default node_label_expression of queue will be used.");
  }

  /**
   */
  public Client() throws Exception  {
    this(new YarnConfiguration());
  }

  /**
   * Helper function to print out usage
   */
  private void printUsage() {
    new HelpFormatter().printHelp("Client", opts);
  }

  /**
   * Parse command line options
   * @param args Parsed command line options 
   * @return Whether the init was successful to run the client
   * @throws ParseException
   */
  public boolean init(String[] args) throws ParseException {

    CommandLine cliParser = new GnuParser().parse(opts, args);

    if (args.length == 0) {
      throw new IllegalArgumentException("No args specified for client to initialize");
    }

    if (cliParser.hasOption("log_properties")) {
      String log4jPath = cliParser.getOptionValue("log_properties");
      try {
        Log4jPropertyHelper.updateLog4jConfiguration(Client.class, log4jPath);
      } catch (Exception e) {
        LOG.warn("Can not set up custom log4j properties. " + e);
      }
    }

    if (cliParser.hasOption("help")) {
      printUsage();
      return false;
    }

    if (cliParser.hasOption("debug")) {
      debugFlag = true;

    }
    
    if  (cliParser.hasOption("enable_tensorboard")) {
      tensorboardFlag = true;
    }

    if (cliParser.hasOption("keep_containers_across_application_attempts")) {
      LOG.info("keep_containers_across_application_attempts");
      keepContainers = true;
    }

    appName = cliParser.getOptionValue("appname", TFConstants.DEFAULT_APPNAME);
    amPriority = Integer.parseInt(cliParser.getOptionValue("priority", "0"));
    amQueue = cliParser.getOptionValue("queue", "default");
    amMemory = Integer.parseInt(cliParser.getOptionValue("master_memory", "10"));		
    amVCores = Integer.parseInt(cliParser.getOptionValue("master_vcores", "1"));

    if (amMemory < 0) {
      throw new IllegalArgumentException("Invalid memory specified for application master, exiting."
          + " Specified memory=" + amMemory);
    }
    if (amVCores < 0) {
      throw new IllegalArgumentException("Invalid virtual cores specified for application master, exiting."
          + " Specified virtual cores=" + amVCores);
    }

    if (!cliParser.hasOption("jar")) {
      throw new IllegalArgumentException("No jar file specified for application master");
    }		

    appMasterJar = cliParser.getOptionValue("jar");

    taskScript = cliParser.getOptionValue("task_script", "");
    taskCmd = cliParser.getOptionValue("task_cmd", "");
    
    if (taskScript.isEmpty() && taskCmd.isEmpty()) {
        throw new IllegalArgumentException(
                "No task script nor task command specified to be executed by application master");        
    }

    if (cliParser.hasOption("task_args")) {
      taskArgs = cliParser.getOptionValues("task_args");
    }
    
    inputPath  = cliParser.getOptionValue("input_path", "");
    outputPath  = cliParser.getOptionValue("output_path", "");
    dockerImage = cliParser.getOptionValue("docker_image", "");

    if (cliParser.hasOption("shell_env")) { 
      String envs[] = cliParser.getOptionValues("shell_env");
      for (String env : envs) {
        env = env.trim();
        int index = env.indexOf('=');
        if (index == -1) {
          shellEnv.put(env, "");
          continue;
        }
        String key = env.substring(0, index);
        String val = "";
        if (index < (env.length()-1)) {
          val = env.substring(index+1);
        }
        shellEnv.put(key, val);
      }
    }

    containerMemory = Integer.parseInt(cliParser.getOptionValue("container_memory", "10"));
    containerVirtualCores = Integer.parseInt(cliParser.getOptionValue("container_vcores", "1"));
    numContainers = cliParser.getOptionValue("num_containers", "");
    
    if (containerMemory < 0 || containerVirtualCores < 0 || numContainers.isEmpty()) {
      throw new IllegalArgumentException("Invalid no. of containers or container memory/vcores specified,"
          + " exiting."
          + " Specified containerMemory=" + containerMemory
          + ", containerVirtualCores=" + containerVirtualCores
          + ", numContainer=" + numContainers);
    }
    
    nodeLabelExpression = cliParser.getOptionValue("node_label_expression", null);

    log4jPropFile = cliParser.getOptionValue("log_properties", "");

    // Get timeline domain options
    if (cliParser.hasOption("domain")) {
      domainId = cliParser.getOptionValue("domain");
      toCreateDomain = cliParser.hasOption("create");
      if (cliParser.hasOption("view_acls")) {
        viewACLs = cliParser.getOptionValue("view_acls");
      }
      if (cliParser.hasOption("modify_acls")) {
        modifyACLs = cliParser.getOptionValue("modify_acls");
      }
    }

    return true;
  }

  /**
   * Main run function for the client
   * @return true if application completed successfully
   * @throws IOException
   * @throws YarnException
   */
  public boolean run() throws IOException, YarnException {

    LOG.info("Running Client");
    yarnClient.start();

    YarnClusterMetrics clusterMetrics = yarnClient.getYarnClusterMetrics();
    LOG.info("Got Cluster metric info from ASM" 
        + ", numNodeManagers=" + clusterMetrics.getNumNodeManagers());

    List<NodeReport> clusterNodeReports = yarnClient.getNodeReports(
        NodeState.RUNNING);
    LOG.info("Got Cluster node info from ASM");
    for (NodeReport node : clusterNodeReports) {
      LOG.info("Got node report from ASM for"
          + ", nodeId=" + node.getNodeId() 
          + ", nodeAddress" + node.getHttpAddress()
          + ", nodeRackName" + node.getRackName()
          + ", nodeNumContainers" + node.getNumContainers());
    }

    QueueInfo queueInfo = yarnClient.getQueueInfo(this.amQueue);
    LOG.info("Queue info"
        + ", queueName=" + queueInfo.getQueueName()
        + ", queueCurrentCapacity=" + queueInfo.getCurrentCapacity()
        + ", queueMaxCapacity=" + queueInfo.getMaximumCapacity()
        + ", queueApplicationCount=" + queueInfo.getApplications().size()
        + ", queueChildQueueCount=" + queueInfo.getChildQueues().size());		

    List<QueueUserACLInfo> listAclInfo = yarnClient.getQueueAclsInfo();
    for (QueueUserACLInfo aclInfo : listAclInfo) {
      for (QueueACL userAcl : aclInfo.getUserAcls()) {
        LOG.info("User ACL Info for Queue"
            + ", queueName=" + aclInfo.getQueueName()			
            + ", userAcl=" + userAcl.name());
      }
    }		

    if (domainId != null && domainId.length() > 0 && toCreateDomain) {
      prepareTimelineDomain();
    }

    // Get a new application id
    YarnClientApplication app = yarnClient.createApplication();
    GetNewApplicationResponse appResponse = app.getNewApplicationResponse();

    int maxMem = appResponse.getMaximumResourceCapability().getMemory();
    LOG.info("Max mem capabililty of resources in this cluster " + maxMem);

    // A resource ask cannot exceed the max. 
    if (amMemory > maxMem) {
      LOG.info("AM memory specified above max threshold of cluster. Using max value."
          + ", specified=" + amMemory
          + ", max=" + maxMem);
      amMemory = maxMem;
    }				

    int maxVCores = appResponse.getMaximumResourceCapability().getVirtualCores();
    LOG.info("Max virtual cores capabililty of resources in this cluster " + maxVCores);
    
    if (amVCores > maxVCores) {
      LOG.info("AM virtual cores specified above max threshold of cluster. " 
          + "Using max value." + ", specified=" + amVCores 
          + ", max=" + maxVCores);
      amVCores = maxVCores;
    }
    
    // set the application name
    ApplicationSubmissionContext appContext = app.getApplicationSubmissionContext();
    ApplicationId appId = appContext.getApplicationId();

    appContext.setKeepContainersAcrossApplicationAttempts(keepContainers);
    appContext.setApplicationName(appName);

    if (attemptFailuresValidityInterval >= 0) {
      appContext
        .setAttemptFailuresValidityInterval(attemptFailuresValidityInterval);
    }

    // set local resources for the application master
    // local files or archives as needed
    // In this scenario, the jar file for the application master is part of the local resources			
    Map<String, LocalResource> localResources = new HashMap<>();

    LOG.info("Copy App Master jar from local filesystem and add to local environment");
    // Copy the application master jar to the filesystem 
    // Create a local resource to point to the destination jar path 
    FileSystem fs = FileSystem.get(conf);
    addToLocalResources(fs, appMasterJar, appMasterJarPath, appId.toString(),
        localResources, null);

    // Set the log4j properties if needed 
    if (!log4jPropFile.isEmpty()) {
      addToLocalResources(fs, log4jPropFile, log4jPath, appId.toString(),
          localResources, null);
    }			

    // The shell script has to be made available on the final container(s)
    // where it will be executed. 
    // To do this, we need to first copy into the filesystem that is visible 
    // to the yarn framework. 
    // We do not need to set this as a local resource for the application 
    // master as the application master does not need it. 		
    String hdfsShellScriptLocation = ""; 
    long hdfsShellScriptLen = 0;
    long hdfsShellScriptTimestamp = 0;
    if (!taskScript.isEmpty()) {
      Path shellSrc = new Path(taskScript);
      String shellPathSuffix =
          appName + "/" + appId.toString() + "/" + SCRIPT_PATH;
      Path shellDst =
          new Path(fs.getHomeDirectory(), shellPathSuffix);
      fs.copyFromLocalFile(false, true, shellSrc, shellDst);
      hdfsShellScriptLocation = shellDst.toUri().toString(); 
      FileStatus shellFileStatus = fs.getFileStatus(shellDst);
      hdfsShellScriptLen = shellFileStatus.getLen();
      hdfsShellScriptTimestamp = shellFileStatus.getModificationTime();
    }

    if (!taskCmd.isEmpty()) {
      addToLocalResources(fs, null, shellCommandPath, appId.toString(),
          localResources, taskCmd);
    }

    if (taskArgs.length > 0) {
      addToLocalResources(fs, null, shellArgsPath, appId.toString(),
          localResources, StringUtils.join(taskArgs, " "));
    }

    // Set the necessary security tokens as needed
    //amContainer.setContainerTokens(containerToken);

    // Set the env variables to be setup in the env where the application master will be run
    LOG.info("Set the environment for the application master");
    Map<String, String> env = new HashMap<String, String>();

    // put location of shell script into env
    // using the env info, the application master will create the correct local resource for the 
    // eventual containers that will be launched to execute the shell scripts
    env.put(TFConstants.SCRIPTLOCATION, hdfsShellScriptLocation);
    env.put(TFConstants.SCRIPTTIMESTAMP, Long.toString(hdfsShellScriptTimestamp));
    env.put(TFConstants.SHELLSCRIPTLEN, Long.toString(hdfsShellScriptLen));
    if (domainId != null && domainId.length() > 0) {
      env.put(TFConstants.TIMELINEDOMAIN, domainId);
    }

    // Add AppMaster.jar location to classpath 		
    // At some point we should not be required to add 
    // the hadoop specific classpaths to the env. 
    // It should be provided out of the box. 
    // For now setting all required classpaths including
    // the classpath to "." for the application jar
    StringBuilder classPathEnv = new StringBuilder(Environment.CLASSPATH.$$())
      .append(ApplicationConstants.CLASS_PATH_SEPARATOR).append("./*");
    for (String c : conf.getStrings(
        YarnConfiguration.YARN_APPLICATION_CLASSPATH,
        YarnConfiguration.DEFAULT_YARN_CROSS_PLATFORM_APPLICATION_CLASSPATH)) {
      classPathEnv.append(ApplicationConstants.CLASS_PATH_SEPARATOR);
      classPathEnv.append(c.trim());
    }
    classPathEnv.append(ApplicationConstants.CLASS_PATH_SEPARATOR).append(
      "./log4j.properties");

    // add the runtime classpath needed for tests to work
    if (conf.getBoolean(YarnConfiguration.IS_MINI_YARN_CLUSTER, false)) {
      classPathEnv.append(':');
      classPathEnv.append(System.getProperty("java.class.path"));
    }

    env.put("CLASSPATH", classPathEnv.toString());

    // Set the necessary command to execute the application master 
    Vector<CharSequence> vargs = new Vector<CharSequence>(30);

    // Set java executable command 
    LOG.info("Setting up app master command");
    vargs.add(Environment.JAVA_HOME.$$() + "/bin/java");
    // Set Xmx based on am memory size
    vargs.add("-Xmx" + amMemory + "m");
    // Set class name 
    vargs.add(appMasterMainClass);
    // Set params for Application Master
    vargs.add("--container_memory " + String.valueOf(containerMemory));
    vargs.add("--container_vcores " + String.valueOf(containerVirtualCores));
    vargs.add("--num_containers " + numContainers);
    vargs.add("--appname " + appName);
    if (!inputPath.isEmpty()) {
      vargs.add("--input_path " + inputPath);
    }
    if (!outputPath.isEmpty()) {
      vargs.add("--output_path " + outputPath);
    }
    if (!dockerImage.isEmpty()) {
      vargs.add("--docker_image " + dockerImage);
    }
    if (null != nodeLabelExpression) {
      appContext.setNodeLabelExpression(nodeLabelExpression);
    }
    vargs.add("--priority " + String.valueOf(shellCmdPriority));

    for (Map.Entry<String, String> entry : shellEnv.entrySet()) {
      vargs.add("--shell_env " + entry.getKey() + "=" + entry.getValue());
    }			
    if (debugFlag) {
      vargs.add("--debug");
    }
    if (tensorboardFlag) {
      vargs.add("--enable_tensorboard");
    }

    vargs.add("1>" + ApplicationConstants.LOG_DIR_EXPANSION_VAR + "/AppMaster.stdout");
    vargs.add("2>" + ApplicationConstants.LOG_DIR_EXPANSION_VAR + "/AppMaster.stderr");

    // Get final commmand
    StringBuilder command = new StringBuilder();
    for (CharSequence str : vargs) {
      command.append(str).append(" ");
    }

    LOG.info("Completed setting up app master command " + command.toString());	   
    List<String> commands = new ArrayList<String>();
    commands.add(command.toString());		

    // Set up the container launch context for the application master
    ContainerLaunchContext amContainer = ContainerLaunchContext.newInstance(
      localResources, env, commands, null, null, null);

    // Set up resource type requirements
    // For now, both memory and vcores are supported, so we set memory and 
    // vcores requirements
    Resource capability = Resource.newInstance(amMemory, amVCores);
    appContext.setResource(capability);

    // Service data is a binary blob that can be passed to the application
    // Not needed in this scenario
    // amContainer.setServiceData(serviceData);

    // Setup security tokens
    if (UserGroupInformation.isSecurityEnabled()) {
      // Note: Credentials class is marked as LimitedPrivate for HDFS and MapReduce
      Credentials credentials = new Credentials();
      String tokenRenewer = conf.get(YarnConfiguration.RM_PRINCIPAL);
      if (tokenRenewer == null || tokenRenewer.length() == 0) {
        throw new IOException(
          "Can't get Master Kerberos principal for the RM to use as renewer");
      }

      // For now, only getting tokens for the default file-system.
      final Token<?> tokens[] =
          fs.addDelegationTokens(tokenRenewer, credentials);
      if (tokens != null) {
        for (Token<?> token : tokens) {
          LOG.info("Got dt for " + fs.getUri() + "; " + token);
        }
      }
      DataOutputBuffer dob = new DataOutputBuffer();
      credentials.writeTokenStorageToStream(dob);
      ByteBuffer fsTokens = ByteBuffer.wrap(dob.getData(), 0, dob.getLength());
      amContainer.setTokens(fsTokens);
    }

    appContext.setAMContainerSpec(amContainer);

    // Set the priority for the application master
    Priority pri = Priority.newInstance(amPriority);
    appContext.setPriority(pri);

    // Set the queue to which this application is to be submitted in the RM
    appContext.setQueue(amQueue);

    // Submit the application to the applications manager
    // SubmitApplicationResponse submitResp = applicationsManager.submitApplication(appRequest);
    // Ignore the response as either a valid response object is returned on success 
    // or an exception thrown to denote some form of a failure
    LOG.info("Submitting application to ASM");

    yarnClient.submitApplication(appContext);

    return true;

  }

  private void addToLocalResources(FileSystem fs, String fileSrcPath,
      String fileDstPath, String appId, Map<String, LocalResource> localResources,
      String resources) throws IOException {
    String suffix =
        appName + "/" + appId + "/" + fileDstPath;
    Path dst =
        new Path(fs.getHomeDirectory(), suffix);
    if (fileSrcPath == null) {
      FSDataOutputStream ostream = null;
      try {
        ostream = FileSystem
            .create(fs, dst, new FsPermission((short) 0710));
        ostream.writeUTF(resources);
      } finally {
        IOUtils.closeQuietly(ostream);
      }
    } else {
      fs.copyFromLocalFile(new Path(fileSrcPath), dst);
    }
    FileStatus scFileStatus = fs.getFileStatus(dst);
    LocalResource scRsrc =
        LocalResource.newInstance(
            ConverterUtils.getYarnUrlFromURI(dst.toUri()),
            LocalResourceType.FILE, LocalResourceVisibility.APPLICATION,
            scFileStatus.getLen(), scFileStatus.getModificationTime());
    localResources.put(fileDstPath, scRsrc);
  }

  private void prepareTimelineDomain() {
    TimelineClient timelineClient = null;
    if (conf.getBoolean(YarnConfiguration.TIMELINE_SERVICE_ENABLED,
        YarnConfiguration.DEFAULT_TIMELINE_SERVICE_ENABLED)) {
      timelineClient = TimelineClient.createTimelineClient();
      timelineClient.init(conf);
      timelineClient.start();
    } else {
      LOG.warn("Cannot put the domain " + domainId +
          " because the timeline service is not enabled");
      return;
    }
    try {
      TimelineDomain domain = new TimelineDomain();
      domain.setId(domainId);
      domain.setReaders(
          viewACLs != null && viewACLs.length() > 0 ? viewACLs : " ");
      domain.setWriters(
          modifyACLs != null && modifyACLs.length() > 0 ? modifyACLs : " ");
      timelineClient.putDomain(domain);
      LOG.info("Put the timeline domain: " +
          TimelineUtils.dumpTimelineRecordtoJSON(domain));
    } catch (Exception e) {
      LOG.error("Error when putting the timeline domain", e);
    } finally {
      timelineClient.stop();
    }
  }
}
