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
    public Color replayColor = new Color(0.4f, 0.7f, 1f);

    private void Update()
    {
        if (orchestrator == null)
            return;

        if (controllerStatusText != null)
        {
            if (orchestrator.IsReplayMode)
            {
                controllerStatusText.text = "Status: Replay Mode";
                controllerStatusText.color = replayColor;
            }
            else
            {
                string progress = "";
                if (orchestrator.BatchTotal > 0)
                    progress = $" ({orchestrator.BatchIndex}/{orchestrator.BatchTotal})";
                controllerStatusText.text = $"Status: {orchestrator.State}{progress}";
                controllerStatusText.color = ColorForState(orchestrator.State);
            }
        }

        if (runConfigText != null)
        {
            if (orchestrator.IsReplayMode)
            {
                runConfigText.text = "RunConfig: Replay Mode";
            }
            else
            {
                string label = string.IsNullOrWhiteSpace(orchestrator.CurrentRunConfigLabel)
                    ? "None"
                    : orchestrator.CurrentRunConfigLabel;
                runConfigText.text = $"RunConfig: {label}";
            }
        }

        if (controllerSelectedText != null)
        {
            if (orchestrator.IsReplayMode)
            {
                controllerSelectedText.text = "Controller: Replay Mode";
            }
            else
            {
                string label = string.IsNullOrWhiteSpace(orchestrator.CurrentControllerLabel)
                    ? "None"
                    : orchestrator.CurrentControllerLabel;
                controllerSelectedText.text = $"Controller: {label}";
            }
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
