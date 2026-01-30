using System;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;
using System.Net.WebSockets;
using System.Threading;

public class UnityControlClient : MonoBehaviour
{
    [Header("Python WebSocket Server")]
    public string host = "127.0.0.1";
    public int port = 7361;

    private ClientWebSocket _ws;
    private CancellationTokenSource _cts;

    public bool IsConnected => _ws != null && _ws.State == WebSocketState.Open;

    private void Awake()
    {
        _cts = new CancellationTokenSource();
    }

    private async void OnDestroy()
    {
        await DisconnectAsync();
        _cts?.Dispose();
    }

    public async Task<bool> TryConnectAsync(bool quiet = true)
    {
        try
        {
            await ConnectAsync(quiet);
            return IsConnected;
        }
        catch
        {
            return false;
        }
    }

    public async Task ConnectAsync(bool quiet = false)
    {
        if (IsConnected) return;

        _ws?.Dispose();
        _ws = new ClientWebSocket();

        var uri = new Uri($"ws://{host}:{port}");

        if (!quiet)
            Debug.Log($"[UnityControlClient] Connecting to {uri} ...");

        await _ws.ConnectAsync(uri, _cts.Token);

        if (!quiet)
            Debug.Log("[UnityControlClient] Connected.");
    }

    public async Task DisconnectAsync()
    {
        if (_ws == null) return;

        try
        {
            if (_ws.State == WebSocketState.Open)
                await _ws.CloseAsync(WebSocketCloseStatus.NormalClosure, "Unity closing", CancellationToken.None);
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[UnityControlClient] Disconnect warning: {e.Message}");
        }
        finally
        {
            _ws.Dispose();
            _ws = null;
        }
    }

    public async Task SendTextAsync(string message, bool readReply = true)
    {
        if (!IsConnected)
            await ConnectAsync(quiet: false);

        byte[] bytes = Encoding.UTF8.GetBytes(message);
        await _ws.SendAsync(new ArraySegment<byte>(bytes), WebSocketMessageType.Text, true, _cts.Token);

        Debug.Log($"[UnityControlClient] Sent {bytes.Length} bytes.");

        if (readReply)
        {
            string reply = await ReceiveTextAsync();
            Debug.Log($"[UnityControlClient] Reply: {reply}");
        }
    }

    public async Task<string> ReceiveTextAsync()
    {
        var buffer = new byte[8192];
        var sb = new StringBuilder();

        while (true)
        {
            var result = await _ws.ReceiveAsync(new ArraySegment<byte>(buffer), _cts.Token);

            if (result.MessageType == WebSocketMessageType.Close)
                return "<server closed>";

            sb.Append(Encoding.UTF8.GetString(buffer, 0, result.Count));

            if (result.EndOfMessage)
                break;
        }

        return sb.ToString();
    }
}
