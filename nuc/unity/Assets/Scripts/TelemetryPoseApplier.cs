using UnityEngine;

public class TelemetryPoseApplier : MonoBehaviour
{
    [Header("References")]
    public UdpTelemetryReceiver receiver;
    public Transform target; // assign your Crazyflie GO transform here

    [Header("Options")]
    public bool applyPosition = true;
    public bool applyRotation = true;
    public bool smooth = true;
    public float positionLerp = 20f;
    public float rotationLerp = 20f;

    private void Reset()
    {
        target = transform;
    }

    private void Update()
    {
        if (receiver == null || target == null)
            return;

        if (!receiver.TryGetLatest(out var pkt))
            return;

        // Position: sim (x,y,z) -> unity (x,z,y)
        Vector3 posUnity = new Vector3(pkt.p[0], pkt.p[2], pkt.p[1]);

        // Rotation:
        // pkt.q is quaternion for sim frame (x,y,z,w). Convert to Unity frame via basis transform.
        // We'll compute R_unity = M * R_sim * M^-1.
        Quaternion qSim = new Quaternion(pkt.q[0], pkt.q[1], pkt.q[2], pkt.q[3]);
        Quaternion qUnity = ConvertSimToUnityRotation(qSim);

        if (applyPosition)
        {
            if (smooth)
                target.position = Vector3.Lerp(target.position, posUnity, 1f - Mathf.Exp(-positionLerp * Time.deltaTime));
            else
                target.position = posUnity;
        }

        if (applyRotation)
        {
            if (smooth)
                target.rotation = Quaternion.Slerp(target.rotation, qUnity, 1f - Mathf.Exp(-rotationLerp * Time.deltaTime));
            else
                target.rotation = qUnity;
        }
    }

    // Axis mapping: sim x->unity x, sim y->unity z, sim z->unity y
    // Use matrix conjugation: R_u = M R_s M^-1
    private static Quaternion ConvertSimToUnityRotation(Quaternion qSim)
    {
        // Build R_sim from qSim
        Matrix4x4 Rsim = Matrix4x4.Rotate(qSim);

        // M maps sim-basis vectors into unity coordinates:
        // unity = [ x_sim, z_sim, y_sim ]
        // So:
        // [ux] = [1 0 0] [sx]
        // [uy] = [0 0 1] [sy]
        // [uz] = [0 1 0] [sz]
        //
        // In matrix form:
        // M = [[1,0,0],
        //      [0,0,1],
        //      [0,1,0]]
        Matrix4x4 M = Matrix4x4.identity;
        M.m00 = 1; M.m01 = 0; M.m02 = 0;
        M.m10 = 0; M.m11 = 0; M.m12 = 1;
        M.m20 = 0; M.m21 = 1; M.m22 = 0;

        Matrix4x4 Minv = M.transpose; // for pure axis swap, inverse == transpose

        Matrix4x4 Runity = M * Rsim * Minv;

        // Extract quaternion
        return Runity.rotation;
    }
}
