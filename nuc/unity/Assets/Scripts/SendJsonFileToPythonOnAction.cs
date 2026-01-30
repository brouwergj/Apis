using System;
using System.IO;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.InputSystem;

public class SendJsonFileToPythonOnAction : MonoBehaviour
{
    [Header("Dependencies")]
    public UnityControlClient controlClient;

    [Header("Input System")]
    [Tooltip("Button action reference to trigger sending.")]
    public InputActionReference sendAction;

    [Header("JSON File")]
    [Tooltip("Filename inside StreamingAssets (e.g., RunConfig.json).")]
    public string fileName = "RunConfig.json";

    [Header("Options")]
    public bool readReply = true;

    private bool _isSending;

    private void OnEnable()
    {
        if (sendAction == null)
        {
            Debug.LogError("[SendJsonFileToPythonOnAction] sendAction is not assigned.");
            return;
        }

        sendAction.action.Enable();
        sendAction.action.performed += OnSendPerformed;
    }

    private void OnDisable()
    {
        if (sendAction == null) return;

        sendAction.action.performed -= OnSendPerformed;
    }

    private async void OnSendPerformed(InputAction.CallbackContext ctx)
    {
        if (_isSending) return;
        _isSending = true;

        try
        {
            await SendJsonFile();
        }
        catch (Exception e)
        {
            Debug.LogError($"[SendJsonFileToPythonOnAction] Send failed: {e}");
        }
        finally
        {
            _isSending = false;
        }
    }

    public async Task SendJsonFile()
    {
        if (controlClient == null)
            throw new InvalidOperationException("controlClient is not assigned.");

        string path = Path.Combine(Application.streamingAssetsPath, fileName);

        if (!File.Exists(path))
            throw new FileNotFoundException($"JSON config file not found: {path}");

        string json = File.ReadAllText(path);

        // Optional sanity check: avoid sending empty payloads
        if (string.IsNullOrWhiteSpace(json))
            throw new InvalidDataException($"JSON file is empty: {path}");

        Debug.Log($"[SendJsonFileToPythonOnAction] Sending {fileName} ({json.Length} chars) ...");
        await controlClient.SendTextAsync(json, readReply: readReply);
    }
}
