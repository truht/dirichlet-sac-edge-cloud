/*
 * Title:        EdgeCloudSim - SimpleJson
 *
 * Description:
 * Minimal JSON writer/reader for the fixed DRL socket protocol
 * (see /PROTOCOL.md at the repo root). Deliberately hand-rolled instead
 * of pulling in an external JSON library: the project has no JSON
 * dependency wired into lib/ or the per-app compile.sh scripts, and
 * the message shapes exchanged with the Python DRL controller are
 * small and fixed (flat objects, numeric arrays, one 2D array). A full
 * general-purpose parser is not needed — this one supports exactly
 * what JSON requires (objects, arrays, strings, numbers, true/false/
 * null) so it stays correct against Python's json.dumps output
 * regardless of whitespace/key ordering.
 *
 * Licence:      GPL - http://www.gnu.org/copyleft/gpl.html
 */

package edu.boun.edgecloudsim.drl;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class SimpleJson {

    private SimpleJson() { }

    // =======================================================================
    // Writing — only the two outgoing message types we need
    // =======================================================================

    public static String writeEpisodeStart(int episode, int numNodes, int stateDim, double[] state) {
        StringBuilder sb = new StringBuilder(64 + state.length * 8);
        sb.append("{\"type\":\"episode_start\",\"episode\":").append(episode)
          .append(",\"num_nodes\":").append(numNodes)
          .append(",\"state_dim\":").append(stateDim)
          .append(",\"state\":");
        appendArray(sb, state);
        sb.append('}');
        return sb.toString();
    }

    public static String writeTransition(int episode, int timeSlot, double[] nextState,
                                          double[] completedLatenciesMs, int dropCount, boolean done,
                                          int cloudRoutedCount, int edgeRoutedCount) {
        StringBuilder sb = new StringBuilder(128 + nextState.length * 8 + completedLatenciesMs.length * 8);
        sb.append("{\"type\":\"transition\",\"episode\":").append(episode)
          .append(",\"time_slot\":").append(timeSlot)
          .append(",\"next_state\":");
        appendArray(sb, nextState);
        sb.append(",\"completed_latencies_ms\":");
        appendArray(sb, completedLatenciesMs);
        sb.append(",\"drop_count\":").append(dropCount)
          .append(",\"cloud_routed_count\":").append(cloudRoutedCount)
          .append(",\"edge_routed_count\":").append(edgeRoutedCount)
          .append(",\"done\":").append(done)
          .append('}');
        return sb.toString();
    }

    private static void appendArray(StringBuilder sb, double[] arr) {
        sb.append('[');
        for (int i = 0; i < arr.length; i++) {
            if (i > 0) sb.append(',');
            appendNumber(sb, arr[i]);
        }
        sb.append(']');
    }

    private static void appendNumber(StringBuilder sb, double d) {
        if (Double.isNaN(d) || Double.isInfinite(d)) {
            sb.append('0'); // JSON has no NaN/Infinity literal; should never legitimately occur
            return;
        }
        if (d == Math.rint(d) && !Double.isInfinite(d) && Math.abs(d) < 1e15) {
            sb.append((long) d);
        } else {
            sb.append(d);
        }
    }

    // =======================================================================
    // Reading — generic-enough JSON parser (object/array/string/number/bool/null)
    // =======================================================================

    /** Parses a single JSON object (the top level of every message on the wire). */
    @SuppressWarnings("unchecked")
    public static Map<String, Object> parseObject(String json) {
        Parser p = new Parser(json);
        p.skipWhitespace();
        Object result = p.parseValue();
        if (!(result instanceof Map)) {
            throw new IllegalArgumentException("Expected top-level JSON object, got: " + json);
        }
        return (Map<String, Object>) result;
    }

    /** Converts a parsed "matrix" field (List<List<Double>>) into a primitive double[][]. */
    @SuppressWarnings("unchecked")
    public static double[][] toMatrix(Object raw) {
        if (!(raw instanceof List)) {
            throw new IllegalArgumentException("Expected 'matrix' to be a JSON array");
        }
        List<Object> rows = (List<Object>) raw;
        double[][] matrix = new double[rows.size()][];
        for (int i = 0; i < rows.size(); i++) {
            List<Object> row = (List<Object>) rows.get(i);
            double[] r = new double[row.size()];
            for (int j = 0; j < row.size(); j++) {
                r[j] = ((Number) row.get(j)).doubleValue();
            }
            matrix[i] = r;
        }
        return matrix;
    }

    public static int asInt(Object raw, int defaultValue) {
        return (raw instanceof Number) ? ((Number) raw).intValue() : defaultValue;
    }

    private static final class Parser {
        private final String s;
        private int pos;

        Parser(String s) {
            this.s = s;
            this.pos = 0;
        }

        void skipWhitespace() {
            while (pos < s.length() && Character.isWhitespace(s.charAt(pos))) pos++;
        }

        char peek() {
            if (pos >= s.length()) {
                throw new IllegalArgumentException("Unexpected end of JSON: " + s);
            }
            return s.charAt(pos);
        }

        Object parseValue() {
            skipWhitespace();
            char c = peek();
            switch (c) {
                case '{': return parseObjectInternal();
                case '[': return parseArrayInternal();
                case '"': return parseString();
                case 't': expect("true"); return Boolean.TRUE;
                case 'f': expect("false"); return Boolean.FALSE;
                case 'n': expect("null"); return null;
                default: return parseNumber();
            }
        }

        Map<String, Object> parseObjectInternal() {
            Map<String, Object> map = new LinkedHashMap<>();
            pos++; // consume '{'
            skipWhitespace();
            if (peek() == '}') { pos++; return map; }
            while (true) {
                skipWhitespace();
                String key = parseString();
                skipWhitespace();
                if (peek() != ':') throw new IllegalArgumentException("Expected ':' at " + pos + " in " + s);
                pos++; // consume ':'
                Object value = parseValue();
                map.put(key, value);
                skipWhitespace();
                char c = peek();
                if (c == ',') { pos++; continue; }
                if (c == '}') { pos++; break; }
                throw new IllegalArgumentException("Expected ',' or '}' at " + pos + " in " + s);
            }
            return map;
        }

        List<Object> parseArrayInternal() {
            List<Object> list = new ArrayList<>();
            pos++; // consume '['
            skipWhitespace();
            if (peek() == ']') { pos++; return list; }
            while (true) {
                list.add(parseValue());
                skipWhitespace();
                char c = peek();
                if (c == ',') { pos++; continue; }
                if (c == ']') { pos++; break; }
                throw new IllegalArgumentException("Expected ',' or ']' at " + pos + " in " + s);
            }
            return list;
        }

        String parseString() {
            if (peek() != '"') throw new IllegalArgumentException("Expected '\"' at " + pos + " in " + s);
            pos++; // consume opening quote
            StringBuilder sb = new StringBuilder();
            while (true) {
                char c = s.charAt(pos++);
                if (c == '"') break;
                if (c == '\\') {
                    char esc = s.charAt(pos++);
                    switch (esc) {
                        case '"': sb.append('"'); break;
                        case '\\': sb.append('\\'); break;
                        case '/': sb.append('/'); break;
                        case 'n': sb.append('\n'); break;
                        case 't': sb.append('\t'); break;
                        case 'r': sb.append('\r'); break;
                        case 'b': sb.append('\b'); break;
                        case 'f': sb.append('\f'); break;
                        case 'u':
                            String hex = s.substring(pos, pos + 4);
                            sb.append((char) Integer.parseInt(hex, 16));
                            pos += 4;
                            break;
                        default: throw new IllegalArgumentException("Invalid escape \\" + esc);
                    }
                } else {
                    sb.append(c);
                }
            }
            return sb.toString();
        }

        Double parseNumber() {
            int start = pos;
            if (peek() == '-') pos++;
            while (pos < s.length() && "0123456789.eE+-".indexOf(s.charAt(pos)) >= 0) pos++;
            return Double.parseDouble(s.substring(start, pos));
        }

        void expect(String literal) {
            if (!s.startsWith(literal, pos)) {
                throw new IllegalArgumentException("Expected '" + literal + "' at " + pos + " in " + s);
            }
            pos += literal.length();
        }
    }
}
