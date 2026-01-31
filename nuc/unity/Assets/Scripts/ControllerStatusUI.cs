using UnityEngine;
using TMPro;

public class ControllerStatusUI : MonoBehaviour
{
    public ControllerOrchestrator orchestrator;
    public TextMeshProUGUI controllerStatusText;
    public TextMeshProUGUI runConfigText;
    public TextMeshProUGUI controllerSelectedText;

    public Color stoppedColor = Color.gray;
    public Color startingColor = Color.yellow;
    public Color readyColor = Color.cyan;
    public Color runningColor = Color.green;
    public Color exitedColor = new Color(1f, 0.5f, 0f);
    public Color errorColor = Color.red;

    private void Update()
    {
        if (orchestrator == null)
            return;

        if (controllerStatusText != null)
        {
            string progress = "";
            if (orchestrator.BatchTotal > 0)
                progress = $" ({orchestrator.BatchIndex}/{orchestrator.BatchTotal})";
            controllerStatusText.text = $"Status: {orchestrator.State}{progress}";
            controllerStatusText.color = ColorForState(orchestrator.State);
        }

        if (runConfigText != null)
        {
            string label = string.IsNullOrWhiteSpace(orchestrator.CurrentRunConfigLabel)
                ? "None"
                : orchestrator.CurrentRunConfigLabel;
            runConfigText.text = $"RunConfig: {label}";
        }

        if (controllerSelectedText != null)
        {
            string label = string.IsNullOrWhiteSpace(orchestrator.CurrentControllerLabel)
                ? "None"
                : orchestrator.CurrentControllerLabel;
            controllerSelectedText.text = $"Controller: {label}";
        }
    }

    private Color ColorForState(ControllerState state)
    {
        return state switch
        {
            ControllerState.Stopped => stoppedColor,
            ControllerState.Starting => startingColor,
            ControllerState.Ready => readyColor,
            ControllerState.Running => runningColor,
            ControllerState.Exited => exitedColor,
            ControllerState.Error => errorColor,
            _ => Color.white
        };
    }
}
