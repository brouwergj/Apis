using System;
using System.Diagnostics;
using System.IO;
using UnityEngine;

public class PythonControllerProcess : MonoBehaviour
{
    [Header("Python")]
    [Tooltip("Absolute path to python executable (e.g. python or python.exe)")]
    public string pythonExe = @"C:\Core\Apis\.venv\Scripts\python.exe";

    [Tooltip("Absolute path to controller.py")]
    public string controllerScriptPath = @"C:\Core\Apis\nuc\python\controller.py";

    [Tooltip("Working directory for the process (optional)")]
    public string workingDirectory = @"C:\Core\Apis\nuc\python";

    [Header("Options")]
    public bool autoStartOnPlay = false;
    public bool logStdout = true;
    public bool logStderr = true;

    private Process _process;
    public event Action OnControllerExited;

    private void Start()
    {
        if (autoStartOnPlay)
            StartController();
    }

    public void StartController()
    {
        if (_process != null && !_process.HasExited)
        {
            UnityEngine.Debug.LogWarning("[PythonControllerProcess] Controller already running.");
            return;
        }

        if (!File.Exists(controllerScriptPath))
        {
            UnityEngine.Debug.LogError($"[PythonControllerProcess] controller.py not found: {controllerScriptPath}");
            return;
        }

        var psi = new ProcessStartInfo
        {
            FileName = pythonExe,
            Arguments = $"\"{controllerScriptPath}\"",
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
        };

        if (!string.IsNullOrEmpty(workingDirectory))
            psi.WorkingDirectory = workingDirectory;

        _process = new Process();
        _process.StartInfo = psi;
        _process.EnableRaisingEvents = true;
        _process.Exited += OnProcessExited;

        if (logStdout)
            _process.OutputDataReceived += (s, e) =>
            {
                if (!string.IsNullOrEmpty(e.Data))
                    UnityEngine.Debug.Log($"[PY] {e.Data}");
            };

        if (logStderr)
            _process.ErrorDataReceived += (s, e) =>
            {
                if (!string.IsNullOrEmpty(e.Data))
                    UnityEngine.Debug.LogError($"[PY] {e.Data}");
            };

        try
        {
            _process.Start();
            _process.BeginOutputReadLine();
            _process.BeginErrorReadLine();

            UnityEngine.Debug.Log("[PythonControllerProcess] Python controller started.");
        }
        catch (Exception e)
        {
            UnityEngine.Debug.LogError($"[PythonControllerProcess] Failed to start process: {e}");
            _process = null;
        }
    }

    public void StopController()
    {
        if (_process == null)
            return;

        try
        {
            if (!_process.HasExited)
            {
                _process.Kill();
                _process.WaitForExit(1000);
            }
        }
        catch (Exception e)
        {
            UnityEngine.Debug.LogWarning($"[PythonControllerProcess] Error stopping process: {e}");
        }
        finally
        {
            _process = null;
            UnityEngine.Debug.Log("[PythonControllerProcess] Python controller stopped.");
        }
    }

    private void OnApplicationQuit()
    {
        StopController();
    }

    private void OnProcessExited(object sender, EventArgs e)
    {
        UnityEngine.Debug.Log("[PythonControllerProcess] Python controller exited.");
        OnControllerExited?.Invoke();
    }

    
}
