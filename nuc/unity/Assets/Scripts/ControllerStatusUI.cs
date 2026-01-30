using UnityEngine;
using TMPro;

public class ControllerStatusUI : MonoBehaviour
{
    public ControllerOrchestrator orchestrator;
    public TextMeshProUGUI text;

    public Color stoppedColor = Color.gray;
    public Color startingColor = Color.yellow;
    public Color readyColor = Color.cyan;
    public Color runningColor = Color.green;
    public Color exitedColor = new Color(1f, 0.5f, 0f);
    public Color errorColor = Color.red;

    private void Update()
    {
        if (orchestrator == null || text == null)
            return;

        text.text = $"Controller: {orchestrator.State}";
        text.color = ColorForState(orchestrator.State);
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
