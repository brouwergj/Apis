using System;
using System.IO;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.UI;
using TMPro;

public class ControllerOrchestrator : MonoBehaviour
{
    [Header("Dependencies")]
    public PythonControllerProcess pythonProcess;
    public UnityControlClient wsClient;
    public LogReplayPlayer replayPlayer;

    [Header("Input Actions")]
    public InputActionReference startControllerAction;   // e.g. UI/StartController
    public InputActionReference startSimAction;          // e.g. UI/StartSimulation

    [Header("RunConfig JSON")]
    private string runConfigsRoot = "";
    private string runConfigFileName = "RunConfig.json";  // relative to runconfigs root
    public bool readReply = true;

    [Header("RunConfig Dropdown (TMP)")]
    public TMP_Dropdown runConfigDropdown;
    public RunConfigDropdownPopulator runConfigPopulator;
    public bool requireUserSelection = true;
    public bool treatPlaceholderAsUnselected = true;
    public string placeholderLabel = "Select RunConfig";

    [Header("Batch Run")]
    public bool runAllConfigs = false;
    public Toggle runAllConfigsToggle;

    [Header("Readiness Polling")]
    public float pollIntervalSeconds = 0.25f;
    public float maxWaitSeconds = 10f;

    public bool IsControllerReady { get; private set; }

    private bool _startingController;
    private bool _startingSim;
    private bool _hasUserSelectedRunConfig;
    private bool _stopRequested;

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
            public string controller;
        }
    }

    public ControllerState State { get; private set; } = ControllerState.Stopped;
    public string LastStatusMessage { get; private set; } = "";
    public string CurrentRunConfigLabel { get; private set; } = "";
    public string CurrentControllerLabel { get; private set; } = "";
    public int BatchIndex { get; private set; } = 0;
    public int BatchTotal { get; private set; } = 0;
    public bool IsReplayMode { get; private set; } = false;

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

        if (runConfigDropdown != null)
        {
            runConfigDropdown.onValueChanged.AddListener(OnRunConfigChanged);
        }

        if (runAllConfigsToggle != null)
        {
            runAllConfigsToggle.onValueChanged.AddListener(OnRunAllToggleChanged);
            runAllConfigs = runAllConfigsToggle.isOn;
        }
    }

    private void OnDisable()
    {
        if (startControllerAction != null)
            startControllerAction.action.performed -= OnStartController;

        if (startSimAction != null)
            startSimAction.action.performed -= OnStartSim;

        if (runConfigDropdown != null)
        {
            runConfigDropdown.onValueChanged.RemoveListener(OnRunConfigChanged);
        }

        if (runAllConfigsToggle != null)
        {
            runAllConfigsToggle.onValueChanged.RemoveListener(OnRunAllToggleChanged);
        }
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
        _stopRequested = false;
        IsReplayMode = false;

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
        _stopRequested = false;
        IsReplayMode = false;

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

        if (_stopRequested)
            return;

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
            if (_stopRequested)
            {
                LastStatusMessage = "Start cancelled";
                State = ControllerState.Stopped;
                return;
            }

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
        await SendRunConfigInternal(bypassSelection: false);
    }

    private async Task SendRunConfigInternal(bool bypassSelection)
    {
        if (_stopRequested)
            return;

        if (!bypassSelection)
        {
            EnsureRunConfigSelection();
            if (runConfigDropdown != null && requireUserSelection && !_hasUserSelectedRunConfig)
            {
                Debug.LogWarning("[ControllerOrchestrator] No RunConfig selected yet; ignoring StartSimulation.");
                LastStatusMessage = "Select a RunConfig first";
                return;
            }
        }

        string path = Path.Combine(ResolveRunConfigsRoot(), runConfigFileName);
        if (!File.Exists(path))
            throw new FileNotFoundException($"RunConfig not found: {path}");

        string json = File.ReadAllText(path);
        if (string.IsNullOrWhiteSpace(json))
            throw new InvalidDataException($"RunConfig is empty: {path}");

        State = ControllerState.Running;
        LastStatusMessage = "Simulation running";
        Debug.Log($"[ControllerOrchestrator] Sending RunConfig {runConfigFileName} ({json.Length} chars)...");

        // Send without consuming replies here; the controller typically sends an immediate ACK
        // and a later DONE message when the episode completes.
        await wsClient.SendTextAsync(json, readReply: false);

        if (readReply)
        {
            // Listen until DONE or socket closes.
            while (true)
            {
                if (_stopRequested)
                {
                    LastStatusMessage = "Stopped";
                    IsControllerReady = false;
                    State = ControllerState.Stopped;
                    await wsClient.DisconnectAsync();
                    break;
                }

                string msg = await wsClient.ReceiveTextAsync();
                Debug.Log($"[ControllerOrchestrator] WS msg: {msg}");

                if (msg == "<server closed>")
                {
                    LastStatusMessage = "Controller socket closed";
                    IsControllerReady = false;
                    State = ControllerState.Exited;
                    await wsClient.DisconnectAsync();
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
                    string controllerName = string.IsNullOrWhiteSpace(reply.summary?.controller)
                        ? "unknown"
                        : reply.summary.controller;
                    LastStatusMessage = $"DONE ({controllerName}). Log: {reply.summary?.log_path}";
                    IsControllerReady = false;
                    State = ControllerState.Exited;
                    await wsClient.DisconnectAsync();
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
        _stopRequested = false;
        IsReplayMode = false;

        _ = StartControllerAndWaitUntilReady()
            .ContinueWith(t =>
            {
                _startingController = false;
            }, TaskScheduler.FromCurrentSynchronizationContext());
    }

    public void UI_StartSimulation()
    {
        if (_startingSim) return;

        _startingSim = true;
        _stopRequested = false;
        IsReplayMode = false;

        _ = StartSimulationAsync()
            .ContinueWith(t =>
            {
                _startingSim = false;
            }, TaskScheduler.FromCurrentSynchronizationContext());
    }

    private async Task StartSimulationAsync()
    {
        bool runAll = runAllConfigsToggle != null ? runAllConfigsToggle.isOn : runAllConfigs;

        if (runAll)
        {
            await RunAllConfigsForCurrentController();
            return;
        }

        EnsureRunConfigSelection();
        if (runConfigDropdown != null && requireUserSelection && !_hasUserSelectedRunConfig)
        {
            Debug.LogWarning("[ControllerOrchestrator] No RunConfig selected yet.");
            LastStatusMessage = "Select a RunConfig first";
            return;
        }

        if (!IsControllerReady)
        {
            Debug.LogWarning("[ControllerOrchestrator] Controller not ready yet.");
            return;
        }

        await SendRunConfigInternal(bypassSelection: false);
    }

    private async Task RunAllConfigsForCurrentController()
    {
        if (runConfigPopulator == null || runConfigPopulator.RunConfigRelativePaths.Length == 0)
        {
            Debug.LogWarning("[ControllerOrchestrator] No RunConfigs available for batch run.");
            LastStatusMessage = "No RunConfigs found";
            return;
        }

        var configs = runConfigPopulator.RunConfigRelativePaths;
        BatchTotal = configs.Length;
        for (int i = 0; i < configs.Length; i++)
        {
            if (_stopRequested)
                break;

            runConfigFileName = configs[i];
            CurrentRunConfigLabel = runConfigFileName;
            _hasUserSelectedRunConfig = true;
            BatchIndex = i + 1;
            LastStatusMessage = $"Running {runConfigFileName} ({BatchIndex}/{BatchTotal})";

            if (!IsControllerReady)
            {
                await WaitForControllerExit();
                await StartControllerAndWaitUntilReady();
            }

            await SendRunConfigInternal(bypassSelection: true);
            await WaitForControllerExit();
        }

        BatchIndex = 0;
        BatchTotal = 0;
    }

    private void OnRunAllToggleChanged(bool isOn)
    {
        runAllConfigs = isOn;
        LastStatusMessage = isOn ? "Run all configs enabled" : "Run all configs disabled";
    }

    private void OnRunConfigChanged(int index)
    {
        if (runConfigDropdown == null)
            return;

        if (index < 0 || index >= runConfigDropdown.options.Count)
            return;

        if (treatPlaceholderAsUnselected &&
            string.Equals(runConfigDropdown.options[index].text, placeholderLabel, StringComparison.Ordinal))
        {
            _hasUserSelectedRunConfig = false;
            return;
        }

        string relPath = runConfigPopulator != null
            ? runConfigPopulator.GetRelativePathForIndex(index)
            : runConfigDropdown.options[index].text;

        if (string.IsNullOrWhiteSpace(relPath))
        {
            _hasUserSelectedRunConfig = false;
            return;
        }

        runConfigFileName = relPath;
        CurrentRunConfigLabel = runConfigFileName;
        _hasUserSelectedRunConfig = true;
        LastStatusMessage = $"RunConfig selected: {runConfigFileName}";
    }

    private void EnsureRunConfigSelection()
    {
        if (runConfigDropdown == null || _hasUserSelectedRunConfig)
            return;

        int index = runConfigDropdown.value;
        if (index < 0 || index >= runConfigDropdown.options.Count)
            return;

        string label = runConfigDropdown.options[index].text;
        if (treatPlaceholderAsUnselected &&
            string.Equals(label, placeholderLabel, StringComparison.Ordinal))
        {
            return;
        }

        string relPath = runConfigPopulator != null
            ? runConfigPopulator.GetRelativePathForIndex(index)
            : label;

        if (string.IsNullOrWhiteSpace(relPath))
            return;

        runConfigFileName = relPath;
        CurrentRunConfigLabel = runConfigFileName;
        _hasUserSelectedRunConfig = true;
        LastStatusMessage = $"RunConfig selected: {runConfigFileName}";
    }

    public void ResetRunConfigSelection(bool updateStatus = true)
    {
        _hasUserSelectedRunConfig = false;
        if (updateStatus)
            LastStatusMessage = "RunConfig list updated";
        CurrentRunConfigLabel = "";
        BatchIndex = 0;
        BatchTotal = 0;
    }

    public void SetSelectedController(string controllerLabel)
    {
        CurrentControllerLabel = controllerLabel ?? "";
        if (!string.IsNullOrWhiteSpace(CurrentControllerLabel))
            LastStatusMessage = $"Controller selected: {CurrentControllerLabel}";
    }

    public void UI_StopAll()
    {
        _ = StopAllAsync();
    }

    public void SetReplayMode(bool isReplayMode)
    {
        IsReplayMode = isReplayMode;
        if (IsReplayMode)
        {
            State = ControllerState.Stopped;
            LastStatusMessage = "Replay Mode";
        }
    }

    private async Task StopAllAsync()
    {
        _stopRequested = true;
        _startingController = false;
        _startingSim = false;

        BatchIndex = 0;
        BatchTotal = 0;
        IsControllerReady = false;
        IsReplayMode = false;
        State = ControllerState.Stopped;
        LastStatusMessage = "Stopped";

        if (runAllConfigsToggle != null)
        {
            runAllConfigsToggle.SetIsOnWithoutNotify(false);
            runAllConfigs = false;
        }
        else
        {
            runAllConfigs = false;
        }

        if (replayPlayer != null)
            replayPlayer.Stop();

        if (wsClient != null)
            await wsClient.DisconnectAsync();

        if (pythonProcess != null)
            pythonProcess.StopController();
    }

    private async Task WaitForControllerExit()
    {
        if (pythonProcess == null)
            return;

        float start = Time.realtimeSinceStartup;
        while (pythonProcess.IsRunning && Time.realtimeSinceStartup - start < maxWaitSeconds)
        {
            await Task.Delay(TimeSpan.FromMilliseconds(50));
        }
    }

    private string ResolveRunConfigsRoot()
    {
        if (!string.IsNullOrWhiteSpace(runConfigsRoot))
            return runConfigsRoot;

        // Assets -> unity -> nuc -> runconfigs
        string path = Path.Combine(Application.dataPath, "..", "..", "runconfigs");
        return Path.GetFullPath(path);
    }

}
