using System;
using System.IO;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.InputSystem;

public class ControllerOrchestrator : MonoBehaviour
{
    [Header("Dependencies")]
    public PythonControllerProcess pythonProcess;
    public UnityControlClient wsClient;

    [Header("Input Actions")]
    public InputActionReference startControllerAction;   // e.g. UI/StartController
    public InputActionReference startSimAction;          // e.g. UI/StartSimulation

    [Header("RunConfig JSON")]
    public string runConfigFileName = "RunConfig.json";  // in StreamingAssets
    public bool readReply = true;

    [Header("Readiness Polling")]
    public float pollIntervalSeconds = 0.25f;
    public float maxWaitSeconds = 10f;

    public bool IsControllerReady { get; private set; }

    private bool _startingController;
    private bool _startingSim;

    [Serializable]
    private class ControllerReply
    {
        public bool ok;
        public string type;      // e.g. "ACK", "DONE"
        public string message;
        public Summary summary;

        [Serializable]
        public class Summary
        {
            public string run_id;
            public string log_path;
            public float dt;
            public float duration_s;
            public int steps_executed;
            public string termination_reason;
            public bool real_time;
        }
    }

    public ControllerState State { get; private set; } = ControllerState.Stopped;
    public string LastStatusMessage { get; private set; } = "";

    private void OnEnable()
    {
        if (startControllerAction != null)
        {
            startControllerAction.action.Enable();
            startControllerAction.action.performed += OnStartController;
        }

        if (startSimAction != null)
        {
            startSimAction.action.Enable();
            startSimAction.action.performed += OnStartSim;
        }
    }

    private void OnDisable()
    {
        if (startControllerAction != null)
            startControllerAction.action.performed -= OnStartController;

        if (startSimAction != null)
            startSimAction.action.performed -= OnStartSim;
    }

    private void Start()
    {
        // Initial state
        State = ControllerState.Stopped;
        LastStatusMessage = "Idle";

        if (pythonProcess != null)
        {
            // Subscribe once
            pythonProcess.OnControllerExited += HandleControllerExited;
        }
    }

    private void OnDestroy()
    {
        // Avoid dangling subscriptions (esp. entering/exiting Play Mode)
        if (pythonProcess != null)
        {
            pythonProcess.OnControllerExited -= HandleControllerExited;
        }
    }

    private void HandleControllerExited()
    {
        IsControllerReady = false;

        // If the process exits while we thought we're running/ready, reflect that.
        State = ControllerState.Exited;
        LastStatusMessage = "Controller process exited";
    }

    private async void OnStartController(InputAction.CallbackContext ctx)
    {
        if (_startingController) return;
        _startingController = true;

        try
        {
            await StartControllerAndWaitUntilReady();
        }
        catch (Exception e)
        {
            Debug.LogError($"[ControllerOrchestrator] StartController failed: {e}");
        }
        finally
        {
            _startingController = false;
        }
    }

    private async void OnStartSim(InputAction.CallbackContext ctx)
    {
        if (_startingSim) return;
        _startingSim = true;

        try
        {
            if (!IsControllerReady)
            {
                Debug.LogWarning("[ControllerOrchestrator] Controller not ready yet; ignoring StartSimulation.");
                return;
            }

            await SendRunConfig();
        }
        catch (Exception e)
        {
            Debug.LogError($"[ControllerOrchestrator] StartSimulation failed: {e}");
        }
        finally
        {
            _startingSim = false;
        }
    }

    public async Task StartControllerAndWaitUntilReady()
    {

        State = ControllerState.Starting;
        LastStatusMessage = "Starting controller process...";

        if (pythonProcess == null || wsClient == null)
            throw new InvalidOperationException("pythonProcess or wsClient not assigned.");

        // Reset readiness
        IsControllerReady = false;

        // Start the python process
        pythonProcess.StartController();

        // Poll websocket until it accepts connections
        Debug.Log("[ControllerOrchestrator] Waiting for controller WebSocket to become available...");

        float start = Time.realtimeSinceStartup;
        while (Time.realtimeSinceStartup - start < maxWaitSeconds)
        {
            bool ok = await wsClient.TryConnectAsync(quiet: true);
            if (ok)
            {
                IsControllerReady = true;
                Debug.Log("[ControllerOrchestrator] Controller is READY (WebSocket connect succeeded).");
                State = ControllerState.Ready;
                LastStatusMessage = "Controller ready (WebSocket connected)";
                return;
            }

            await Task.Delay(TimeSpan.FromSeconds(pollIntervalSeconds));
        }

        State = ControllerState.Error;
        LastStatusMessage = "Timed out waiting for controller";
        Debug.LogError("[ControllerOrchestrator] Timed out waiting for controller WebSocket.");
    }

    public async Task SendRunConfig()
    {
        string path = Path.Combine(Application.streamingAssetsPath, runConfigFileName);
        if (!File.Exists(path))
            throw new FileNotFoundException($"RunConfig not found: {path}");

        string json = File.ReadAllText(path);
        if (string.IsNullOrWhiteSpace(json))
            throw new InvalidDataException($"RunConfig is empty: {path}");

        State = ControllerState.Running;
        LastStatusMessage = "Simulation running";
        Debug.Log($"[ControllerOrchestrator] Sending RunConfig ({json.Length} chars)...");

        // Send without consuming replies here; the controller typically sends an immediate ACK
        // and a later DONE message when the episode completes.
        await wsClient.SendTextAsync(json, readReply: false);

        if (readReply)
        {
            // Listen until DONE or socket closes.
            while (true)
            {
                string msg = await wsClient.ReceiveTextAsync();
                Debug.Log($"[ControllerOrchestrator] WS msg: {msg}");

                if (msg == "<server closed>")
                {
                    LastStatusMessage = "Controller socket closed";
                    break;
                }

                ControllerReply reply = null;
                try
                {
                    reply = JsonUtility.FromJson<ControllerReply>(msg);
                }
                catch
                {
                    // If it isn't JSON we understand, show it as a status line and keep listening.
                    LastStatusMessage = msg;
                    continue;
                }

                if (!string.IsNullOrEmpty(reply.message))
                    LastStatusMessage = reply.message;

                if (reply.type == "DONE")
                {
                    LastStatusMessage = $"DONE. Log: {reply.summary?.log_path}";
                    // The Python controller exits after DONE in your current design; process-exit
                    // will be handled via OnControllerExited.
                    break;
                }
            }
        }

        // NOTE: your current python server exits after handling one START.
        // After it exits, the websocket will close and IsControllerReady will become stale.
        // We'll handle that properly when we make python persistent.
    }

    public void UI_StartController()
    {
        if (_startingController) return;
        _startingController = true;

        _ = StartControllerAndWaitUntilReady()
            .ContinueWith(t =>
            {
                _startingController = false;
            }, TaskScheduler.FromCurrentSynchronizationContext());
    }

    public void UI_StartSimulation()
    {
        if (_startingSim) return;

        if (!IsControllerReady)
        {
            Debug.LogWarning("[ControllerOrchestrator] Controller not ready yet.");
            return;
        }

        _startingSim = true;

        _ = SendRunConfig()
            .ContinueWith(t =>
            {
                _startingSim = false;
            }, TaskScheduler.FromCurrentSynchronizationContext());
    }

}
