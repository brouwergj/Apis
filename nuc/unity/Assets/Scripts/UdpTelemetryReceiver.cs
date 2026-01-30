using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

public class UdpTelemetryReceiver : MonoBehaviour
{
    [Header("UDP")]
    public int listenPort = 15000;

    [Header("Debug")]
    public bool logPackets = false;

    // Latest-only snapshot (thread-safe via lock)
    private readonly object _lock = new object();
    private TelemetryPacket _latest;
    private bool _hasLatest = false;

    private UdpClient _client;
    private Thread _thread;
    private volatile bool _running;

    [Serializable]
    public class TelemetryPacket
    {
        public float t;
        public float[] p; // 3
        public float[] q; // 4 (x,y,z,w)
        public float[] v; // 3 (optional)

        public float thrust;       // NEW
        public float thrust_norm;  // NEW
    }

    private void OnEnable()
    {
        StartReceiver();
    }

    private void OnDisable()
    {
        StopReceiver();
    }

    private void OnDestroy()
    {
        StopReceiver();
    }

    public bool TryGetLatest(out TelemetryPacket pkt)
    {
        lock (_lock)
        {
            if (!_hasLatest)
            {
                pkt = null;
                return false;
            }

            // Return reference; caller should read immediately
            pkt = _latest;
            return true;
        }
    }

    private void StartReceiver()
    {
        if (_running) return;

        _client = new UdpClient(listenPort);
        _client.Client.ReceiveTimeout = 500; // ms (so thread can exit promptly)
        _running = true;

        _thread = new Thread(ReceiveLoop);
        _thread.IsBackground = true;
        _thread.Start();

        Debug.Log($"[UdpTelemetryReceiver] Listening on UDP :{listenPort}");
    }

    private void StopReceiver()
    {
        _running = false;

        try
        {
            _client?.Close();
        }
        catch { /* ignore */ }

        _client = null;

        try
        {
            if (_thread != null && _thread.IsAlive)
                _thread.Join(1000);
        }
        catch { /* ignore */ }

        _thread = null;
    }

    private void ReceiveLoop()
    {
        var any = new IPEndPoint(IPAddress.Any, 0);

        while (_running)
        {
            try
            {
                byte[] data = _client.Receive(ref any); // blocks until timeout or packet
                if (data == null || data.Length == 0) continue;

                string json = Encoding.UTF8.GetString(data);

                if (logPackets)
                    Debug.Log($"[UdpTelemetryReceiver] {json}");

                // JsonUtility is fast and fine for this payload shape
                var pkt = JsonUtility.FromJson<TelemetryPacket>(json);
                if (pkt == null || pkt.p == null || pkt.q == null || pkt.p.Length < 3 || pkt.q.Length < 4)
                    continue;

                // Latest-only: overwrite
                lock (_lock)
                {
                    _latest = pkt;
                    _hasLatest = true;
                }
            }
            catch (SocketException)
            {
                // timeout or socket closed — loop continues or exits based on _running
            }
            catch (ObjectDisposedException)
            {
                // socket closed during shutdown
                break;
            }
            catch (Exception)
            {
                // swallow to keep receiver alive (you can log if you want)
            }
        }
    }
}
