using System;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.InputSystem;

public class SendToPythonOnAction : MonoBehaviour
{
    [Header("Dependencies")]
    public UnityControlClient controlClient;

    [Header("Input System")]
    [Tooltip("Reference to a Button action in your Input Actions asset (e.g., UI/SendToPython).")]
    public InputActionReference sendAction;

    [Header("Message Payload")]
    [TextArea(3, 10)]
    public string jsonMessage = "{ \"type\": \"START\", \"run_id\": \"demo_001\", \"seed\": 42 }";

    private bool _isSending;

    private void OnEnable()
    {
        if (sendAction == null)
        {
            Debug.LogError("[SendToPythonOnAction] sendAction is not assigned.");
            return;
        }

        // Ensure enabled and subscribe
        sendAction.action.Enable();
        sendAction.action.performed += OnSendPerformed;
    }

    private void OnDisable()
    {
        if (sendAction == null) return;

        sendAction.action.performed -= OnSendPerformed;
        // Optional: disable if you want the action inactive when this component is disabled
        // sendAction.action.Disable();
    }

    private async void OnSendPerformed(InputAction.CallbackContext ctx)
    {
        // Button actions can fire multiple times depending on interactions.
        // This keeps it simple: ignore while already sending.
        if (_isSending) return;
        _isSending = true;

        try
        {
            await SendOnce();
        }
        catch (Exception e)
        {
            Debug.LogError($"[SendToPythonOnAction] Send failed: {e}");
        }
        finally
        {
            _isSending = false;
        }
    }

    public async Task SendOnce()
    {
        if (controlClient == null)
        {
            Debug.LogError("[SendToPythonOnAction] controlClient not assigned.");
            return;
        }

        await controlClient.SendTextAsync(jsonMessage, readReply: true);
    }
}
