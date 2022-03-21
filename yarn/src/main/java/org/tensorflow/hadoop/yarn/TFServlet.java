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
import java.io.PrintWriter;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

public class TFServlet extends HttpServlet
{
    private static final long serialVersionUID = 7965676366699736489L;

    @Override
    public void doGet(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException
    {
        ApplicationMaster applicationMaster = (ApplicationMaster) getServletContext().getAttribute(ApplicationMaster.class.getName());
        PrintWriter out = response.getWriter();
        response.setContentType("text/html");
        applicationMaster.printHtmlStatus(out);
        out.close();
    }
}
