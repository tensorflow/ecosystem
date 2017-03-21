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
 */

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.registry.client.api.RegistryConstants;
import org.apache.hadoop.util.JarFinder;
import org.apache.hadoop.yarn.api.records.ApplicationReport;
import org.apache.hadoop.yarn.api.records.FinalApplicationStatus;
import org.apache.hadoop.yarn.api.records.YarnApplicationState;
import org.apache.hadoop.yarn.api.records.timeline.TimelineEntities;
import org.apache.hadoop.yarn.api.records.timeline.TimelineEntity;
import org.apache.hadoop.yarn.api.records.timeline.TimelineEvent;
import org.apache.hadoop.yarn.client.api.YarnClient;
import org.apache.hadoop.yarn.conf.YarnConfiguration;
import org.apache.hadoop.yarn.server.MiniYARNCluster;
import org.apache.hadoop.yarn.server.resourcemanager.scheduler.capacity.CapacityScheduler;
import org.junit.*;

import java.io.*;
import java.net.URL;
import java.util.List;

import static java.lang.Thread.sleep;

public class TestTF {
    private static final Log LOG = LogFactory.getLog(TestTF.class);

    private static MiniYARNCluster yarnCluster = null;
    private static YarnConfiguration conf = null;

    // Internal ZooKeeper instance for integration test run
    private static EmbeddedZKServer zkServer;

    private final static String APPMASTER_JAR = JarFinder.getJar(ApplicationMaster.class);
    private String envCheckShell;

    private static synchronized File createYarnSiteConfig(Configuration yarn_conf) throws IOException {
        URL url = Thread.currentThread().getContextClassLoader().getResource("yarn-site.xml");
        if (url == null) {
            throw new RuntimeException("Could not find 'yarn-site.xml' dummy file in classpath");
        }
        File yarnSiteXml = new File(url.getPath());
        FileWriter writer = new FileWriter(yarnSiteXml);
        yarn_conf.set("yarn.application.classpath", yarnSiteXml.getParent());
        yarn_conf.setInt("yarn.nodemanager.delete.debug-delay-sec", 600);
        yarn_conf.writeXml(writer);
        writer.flush();
        writer.close();
        return yarnSiteXml;
    }

    @Before
    public void setup() throws Exception {
        LOG.info("Starting up YARN cluster");

        if (zkServer == null) {
            zkServer = new EmbeddedZKServer();
            zkServer.start();
        }

        conf = new YarnConfiguration();
        conf.setInt(YarnConfiguration.RM_SCHEDULER_MINIMUM_ALLOCATION_MB, 1024);
        conf.set("yarn.log.dir", "target");
        conf.setBoolean(YarnConfiguration.TIMELINE_SERVICE_ENABLED, true);
        conf.set(RegistryConstants.KEY_REGISTRY_ZK_QUORUM, "localhost:" + zkServer.port());
        conf.set(YarnConfiguration.RM_SCHEDULER, CapacityScheduler.class.getName());

        if (yarnCluster == null) {
            yarnCluster = new MiniYARNCluster(
                    TestTF.class.getSimpleName(), 1, 1, 1, 1);
            yarnCluster.init(conf);

            yarnCluster.start();
            conf.set(YarnConfiguration.TIMELINE_SERVICE_WEBAPP_ADDRESS,
                    MiniYARNCluster.getHostname() + ":"
                            + yarnCluster.getApplicationHistoryServer().getPort());
        }
        sleep(2000);

        Configuration miniyarn_conf = yarnCluster.getConfig();
        createYarnSiteConfig(miniyarn_conf);

        URL url = Thread.currentThread().getContextClassLoader().getResource("env_check.sh");
        if (url == null) {
            throw new RuntimeException("Could not find 'env_check.sh' file in resources");
        }
        envCheckShell = url.getPath();

    }

    @After
    public void tearDown() throws IOException {
        if (yarnCluster != null) {
            LOG.info("shutdown MiniYarn cluster");
            try {
                yarnCluster.stop();
            } finally {
                yarnCluster = null;
            }
        }

        //shutdown Zookeeper server
        if (zkServer != null) {
            LOG.info("shutdown zookeeper");
            zkServer.stop();
            zkServer = null;
        }
    }

    /*
     * Launches the client, waits for the app to finish and returns the report
    */
    private FinalApplicationStatus getApplicationReport(String[] args) throws Exception {
        final Client client = new Client(new Configuration(yarnCluster.getConfig()));
        LOG.info("Initializing YARN TensorFlow Client");
        boolean initSuccess = client.init(args);
        Assert.assertTrue(initSuccess);
        LOG.info("Running YARN TensorFlow Client");
        boolean result = client.run();
        Assert.assertTrue(result);
        LOG.info("Client run completed.");
        YarnClient yarnClient = YarnClient.createYarnClient();
        yarnClient.init(new Configuration(yarnCluster.getConfig()));
        yarnClient.start();
        ApplicationReport appReport = null;
        boolean finished = false;
        YarnApplicationState state = YarnApplicationState.NEW;
        while (!finished) {
            List<ApplicationReport> apps = yarnClient.getApplications();
            if (apps.size() == 0) {
                sleep(10);
                continue;
            }
            appReport = apps.get(0);
            if (appReport.getHost().equals("N/A")) {
                sleep(10);
                continue;
            }
            state = appReport.getYarnApplicationState();
            if (state == YarnApplicationState.FINISHED || state == YarnApplicationState.FAILED || state == YarnApplicationState.KILLED) {
                finished = true;
            }
        }
        Assert.assertNotNull(appReport);
        Assert.assertEquals(state, YarnApplicationState.FINISHED);
        return appReport.getFinalApplicationStatus();
    }

    /*
     * Launching 3 containers, 1 ps and 2 workers, all successful
     */
    @Test(timeout = 90000)
    public void testPositive() throws Exception {
        String[] args = {
                "--jar",
                APPMASTER_JAR,
                "-container_vcores",
                "1",
                "-container_memory",
                "1024",
                "-num_containers",
                "ps:1,worker:2",
                "-input_path", ".",
                "-output_path", ".",
                "-task_script", envCheckShell,
                "-task_cmd", "sh ${DTF_TASK_SCRIPT}",
                "-task_args", "test"
        };

        FinalApplicationStatus status = getApplicationReport(args);
        Assert.assertEquals(FinalApplicationStatus.SUCCEEDED, status);

        TimelineEntities entitiesAttempts = yarnCluster
                .getApplicationHistoryServer()
                .getTimelineStore()
                .getEntities(ApplicationMaster.TFEntity.TF_APP_ATTEMPT.toString(),
                        null, null, null, null, null, null, null, null, null);
        Assert.assertNotNull(entitiesAttempts);
        Assert.assertEquals(1, entitiesAttempts.getEntities().size());
        Assert.assertEquals(2, entitiesAttempts.getEntities().get(0).getEvents()
                .size());
        Assert.assertEquals(entitiesAttempts.getEntities().get(0).getEntityType()
                , ApplicationMaster.TFEntity.TF_APP_ATTEMPT.toString());
        TimelineEntities entities = yarnCluster
                .getApplicationHistoryServer()
                .getTimelineStore()
                .getEntities(ApplicationMaster.TFEntity.TF_CONTAINER.toString(), null,
                        null, null, null, null, null, null, null, null);
        Assert.assertNotNull(entities);
        // "ps:1,worker:2" = 3 containers
        Assert.assertEquals(3, entities.getEntities().size());
        Assert.assertEquals(entities.getEntities().get(0).getEntityType(), ApplicationMaster.TFEntity.TF_CONTAINER.toString());
        for (TimelineEntity entity: entities.getEntities()) {
            // There are TF_CONTAINER_START and TF_CONTAINER_END events
            for (TimelineEvent event: entity.getEvents()) {
                if (event.getEventType().equals(ApplicationMaster.TFEvent.TF_CONTAINER_END.toString())) {
                    int exitStatus = (Integer) event.getEventInfo().get(ApplicationMaster.TFInfo.TF_EXIT_STATUS.toString());
                    Assert.assertEquals(0, exitStatus);
                }
            }
        }
    }


    /*
     * Launching 3 containers, 1 ps and 2 workers, all fails
     */
    @Test(timeout = 90000)
    public void testFailAll() throws Exception {
        String[] args = {
                "--jar",
                APPMASTER_JAR,
                "-container_vcores",
                "1",
                "-container_memory",
                "1024",
                "-num_containers",
                "ps:1,worker:2",
                "-input_path", ".",
                "-output_path", ".",
                "-task_script", envCheckShell,
                "-task_cmd", "sh ${DTF_TASK_SCRIPT} && exit 1",
                "-task_args", ""
        };

        FinalApplicationStatus status = getApplicationReport(args);

        Assert.assertEquals(FinalApplicationStatus.FAILED, status);

        TimelineEntities entitiesAttempts = yarnCluster
                .getApplicationHistoryServer()
                .getTimelineStore()
                .getEntities(ApplicationMaster.TFEntity.TF_APP_ATTEMPT.toString(),
                        null, null, null, null, null, null, null, null, null);
        Assert.assertNotNull(entitiesAttempts);
        Assert.assertEquals(1, entitiesAttempts.getEntities().size());
        Assert.assertEquals(2, entitiesAttempts.getEntities().get(0).getEvents()
                .size());
        Assert.assertEquals(entitiesAttempts.getEntities().get(0).getEntityType()
                , ApplicationMaster.TFEntity.TF_APP_ATTEMPT.toString());

        TimelineEntities entities = yarnCluster
                .getApplicationHistoryServer()
                .getTimelineStore()
                .getEntities(ApplicationMaster.TFEntity.TF_CONTAINER.toString(), null,
                        null, null, null, null, null, null, null, null);
        Assert.assertNotNull(entities);
        // "ps:1,worker:2" = 3 containers
        Assert.assertEquals(3, entities.getEntities().size());
        Assert.assertEquals(entities.getEntities().get(0).getEntityType(), ApplicationMaster.TFEntity.TF_CONTAINER.toString());
        for (TimelineEntity entity: entities.getEntities()) {
            // There are TF_CONTAINER_START and TF_CONTAINER_END events
            for (TimelineEvent event: entity.getEvents()) {
                if (event.getEventType().equals(ApplicationMaster.TFEvent.TF_CONTAINER_END.toString())) {
                    int exitStatus = (Integer) event.getEventInfo().get(ApplicationMaster.TFInfo.TF_EXIT_STATUS.toString());
                    Assert.assertEquals(1, exitStatus);
                }
            }
        }
    }

    /*
     * Launching 3 containers, 1 ps and 2 workers, fail the chief worker (task index 0)
     */
    @Test(timeout = 90000)
    public void testFailChiefWorker() throws Exception {
        String[] args = {
                "--jar",
                APPMASTER_JAR,
                "-container_vcores",
                "1",
                "-container_memory",
                "1024",
                "-num_containers",
                "ps:1,worker:2",
                "-input_path", ".",
                "-output_path", ".",
                "-task_script", envCheckShell,
                "-task_cmd", "sh ${DTF_TASK_SCRIPT} && if [ \"${DTF_TASK_JOB_NAME}\" == 'worker' -a \"${DTF_TASK_INDEX}\" == '0' ]; then exit 1; fi",
                "-task_args", ""
        };

        FinalApplicationStatus status = getApplicationReport(args);

        Assert.assertEquals(FinalApplicationStatus.FAILED, status);

        TimelineEntities entitiesAttempts = yarnCluster
                .getApplicationHistoryServer()
                .getTimelineStore()
                .getEntities(ApplicationMaster.TFEntity.TF_APP_ATTEMPT.toString(),
                        null, null, null, null, null, null, null, null, null);
        Assert.assertNotNull(entitiesAttempts);
        Assert.assertEquals(1, entitiesAttempts.getEntities().size());
        Assert.assertEquals(2, entitiesAttempts.getEntities().get(0).getEvents()
                .size());
        Assert.assertEquals(entitiesAttempts.getEntities().get(0).getEntityType()
                , ApplicationMaster.TFEntity.TF_APP_ATTEMPT.toString());

        TimelineEntities entities = yarnCluster
                .getApplicationHistoryServer()
                .getTimelineStore()
                .getEntities(ApplicationMaster.TFEntity.TF_CONTAINER.toString(), null,
                        null, null, null, null, null, null, null, null);
        Assert.assertNotNull(entities);
        // "ps:1,worker:2" = 3 containers
        Assert.assertEquals(3, entities.getEntities().size());
        Assert.assertEquals(entities.getEntities().get(0).getEntityType(), ApplicationMaster.TFEntity.TF_CONTAINER.toString());
        for (TimelineEntity entity: entities.getEntities()) {
            // There are TF_CONTAINER_START and TF_CONTAINER_END events
            for (TimelineEvent event: entity.getEvents()) {
                if (event.getEventType().equals(ApplicationMaster.TFEvent.TF_CONTAINER_END.toString())) {
                    int exitStatus = (Integer) event.getEventInfo().get(ApplicationMaster.TFInfo.TF_EXIT_STATUS.toString());
                    String taskName = (String) event.getEventInfo().get(ApplicationMaster.TFInfo.TF_TASK_NAME.toString());
                    if (taskName.equals("worker[0]")) {
                        Assert.assertEquals(1, exitStatus);
                    } else {
                        Assert.assertEquals(0, exitStatus);
                    }
                }
            }
        }
    }

    /*
    * Launching 3 containers, 1 ps and 2 workers, fail the non-chief worker (task index 1)
    */
    @Test(timeout = 90000)
    public void testFailNonChiefWorker() throws Exception {
        String[] args = {
                "--jar",
                APPMASTER_JAR,
                "-container_vcores",
                "1",
                "-container_memory",
                "1024",
                "-num_containers",
                "ps:1,worker:2",
                "-input_path", ".",
                "-output_path", ".",
                "-task_script", envCheckShell,
                "-task_cmd", "sh ${DTF_TASK_SCRIPT} && if [ ${DTF_TASK_JOB_NAME} == 'worker' -a ${DTF_TASK_INDEX} == '1' ]; then exit 1; fi",
                "-task_args", ""
        };

        FinalApplicationStatus status = getApplicationReport(args);

        Assert.assertEquals(FinalApplicationStatus.FAILED, status);

        TimelineEntities entitiesAttempts = yarnCluster
                .getApplicationHistoryServer()
                .getTimelineStore()
                .getEntities(ApplicationMaster.TFEntity.TF_APP_ATTEMPT.toString(),
                        null, null, null, null, null, null, null, null, null);
        Assert.assertNotNull(entitiesAttempts);
        Assert.assertEquals(1, entitiesAttempts.getEntities().size());
        Assert.assertEquals(2, entitiesAttempts.getEntities().get(0).getEvents()
                .size());
        Assert.assertEquals(entitiesAttempts.getEntities().get(0).getEntityType()
                , ApplicationMaster.TFEntity.TF_APP_ATTEMPT.toString());

        TimelineEntities entities = yarnCluster
                .getApplicationHistoryServer()
                .getTimelineStore()
                .getEntities(ApplicationMaster.TFEntity.TF_CONTAINER.toString(), null,
                        null, null, null, null, null, null, null, null);
        Assert.assertNotNull(entities);
        // "ps:1,worker:2" = 3 containers
        Assert.assertEquals(3, entities.getEntities().size());
        Assert.assertEquals(entities.getEntities().get(0).getEntityType(), ApplicationMaster.TFEntity.TF_CONTAINER.toString());
        for (TimelineEntity entity: entities.getEntities()) {
            // There are TF_CONTAINER_START and TF_CONTAINER_END events
            for (TimelineEvent event: entity.getEvents()) {
                if (event.getEventType().equals(ApplicationMaster.TFEvent.TF_CONTAINER_END.toString())) {
                    int exitStatus = (Integer) event.getEventInfo().get(ApplicationMaster.TFInfo.TF_EXIT_STATUS.toString());
                    String taskName = (String) event.getEventInfo().get(ApplicationMaster.TFInfo.TF_TASK_NAME.toString());
                    if (taskName.equals("worker[1]")) {
                        Assert.assertEquals(1, exitStatus);
                    } else {
                        Assert.assertEquals(0, exitStatus);
                    }
                }
            }
        }
    }
}
