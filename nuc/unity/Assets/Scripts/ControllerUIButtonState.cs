using UnityEngine;
using UnityEngine.UI;

public class ControllerUIButtonState : MonoBehaviour
{
    public ControllerOrchestrator orchestrator;
    public Button startControllerButton;
    public Button startSimulationButton;

    private void Update()
    {
        if (orchestrator == null) return;

        startControllerButton.interactable =
            orchestrator.State == ControllerState.Stopped ||
            orchestrator.State == ControllerState.Exited;

        startSimulationButton.interactable =
            orchestrator.State == ControllerState.Ready;
    }
}
