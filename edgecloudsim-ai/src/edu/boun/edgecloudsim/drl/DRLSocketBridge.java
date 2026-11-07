/*
 * Title:        EdgeCloudSim - DRLSocketBridge
 *
 * Description:
 * TCP server-side bridge to the Central DRL Coordinator (Python client).
 * Implements the wire protocol defined in /PROTOCOL.md: Java opens the
 * ServerSocket and accepts exactly once, before the episode loop starts,
 * then reuses the same connection for every episode of training.
 *
 * Licence:      GPL - http://www.gnu.org/copyleft/gpl.html
 */

package edu.boun.edgecloudsim.drl;

import java.io.BufferedReader;
import java.io.Closeable;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.Map;

/**
 * Server-side half of the DRL &lt;-&gt; EdgeCloudSim socket protocol.
 *
 * <p>Lifecycle (see /PROTOCOL.md section 5): construct once, call
 * {@link #waitForClient()} once before the {@code Max_Episodes} loop,
 * then call {@link #sendEpisodeStart}/{@link #sendTransition} and
 * {@link #receiveActionOrNullOnShutdown()} once per time-slot for the
 * entire training run. Close only after all episodes finish.</p>
 */
public class DRLSocketBridge implements Closeable {

    private final ServerSocket serverSocket;
    private Socket clientSocket;
    private BufferedReader in;
    private PrintWriter out;

    public DRLSocketBridge(int port) throws IOException {
        this.serverSocket = new ServerSocket(port);
    }

    /** Blocks until the Python client connects. Call exactly once, before the episode loop. */
    public void waitForClient() throws IOException {
        System.out.println("[DRLSocketBridge] Waiting for Python DRL client on port "
                + serverSocket.getLocalPort() + " ...");
        clientSocket = serverSocket.accept();
        in = new BufferedReader(new InputStreamReader(clientSocket.getInputStream(), StandardCharsets.UTF_8));
        out = new PrintWriter(new OutputStreamWriter(clientSocket.getOutputStream(), StandardCharsets.UTF_8), true);
        System.out.println("[DRLSocketBridge] Python client connected: " + clientSocket.getRemoteSocketAddress());
    }

    public void sendEpisodeStart(int episode, int numNodes, int stateDim, double[] state) throws IOException {
        send(SimpleJson.writeEpisodeStart(episode, numNodes, stateDim, state));
    }

    public void sendTransition(int episode, int timeSlot, double[] nextState,
                                double[] completedLatenciesMs, int dropCount, boolean done,
                                int cloudRoutedCount, int edgeRoutedCount) throws IOException {
        send(SimpleJson.writeTransition(episode, timeSlot, nextState, completedLatenciesMs, dropCount, done,
                cloudRoutedCount, edgeRoutedCount));
    }

    private void send(String json) throws IOException {
        out.print(json);
        out.print('\n');
        out.flush();
        if (out.checkError()) {
            throw new IOException("Failed to write to DRL socket (Python client may have disconnected)");
        }
    }

    /**
     * Blocks until the next message from Python arrives.
     *
     * @return the parsed N x (N+1) action matrix, or {@code null} if Python sent
     *         a {@code shutdown} message (training finished — caller should stop).
     */
    public double[][] receiveActionOrNullOnShutdown() throws IOException {
        String line = in.readLine();
        if (line == null) {
            throw new EOFException("DRL socket closed by Python peer unexpectedly");
        }
        Map<String, Object> msg = SimpleJson.parseObject(line);
        Object type = msg.get("type");
        if ("shutdown".equals(type)) {
            return null;
        }
        if (!"action".equals(type)) {
            throw new IOException("Expected 'action' or 'shutdown' message, got: " + type);
        }
        return SimpleJson.toMatrix(msg.get("matrix"));
    }

    @Override
    public void close() throws IOException {
        try {
            if (clientSocket != null) clientSocket.close();
        } finally {
            serverSocket.close();
        }
    }
}
