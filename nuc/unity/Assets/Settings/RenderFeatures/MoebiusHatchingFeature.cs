using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.RenderGraphModule.Util; // <-- AddBlitPass helpers

public class MoebiusHatchingFeature : ScriptableRendererFeature
{
    [System.Serializable]
    public class HatchingSettings
    {
        public RenderPassEvent renderPassEvent = RenderPassEvent.AfterRenderingTransparents;

        [Header("Material")]
        public Material material;

        [Header("Requirements (enable on URP Asset/Camera)")]
        public bool requireDepthTexture = true;
        public bool requireNormalsTexture = false;

        [Header("Edge")]
        [Range(0, 3.0f)] public float edgeThickness = 1.0f;
        [Range(0f, 2f)]     public float edgeIntensity = 1.0f;

        [Header("Hatching")]
        [Range(0.5f, 125f)]   public float hatchScale = 1.5f;
        [Range(0f, 1f)]     public float hatchContrast = 0.65f;
        [ColorUsage(false, true)] public Color inkColor   = new(0.05f, 0.04f, 0.03f, 1f);
        [ColorUsage(false, true)] public Color paperColor = new(0.98f, 0.97f, 0.92f, 1f);

        [Header("Mix")]
        [Range(0f, 1f)] public float edgeToonMix    = 0.9f;
        [Range(0f, 1f)] public float sceneInfluence = 0.25f;
    }

    class HatchingPass : ScriptableRenderPass
    {
        private readonly string m_ProfilerTag;
        private readonly HatchingSettings m_Settings;

        static readonly int _EdgeThicknessID  = Shader.PropertyToID("_EdgeThickness");
        static readonly int _EdgeIntensityID  = Shader.PropertyToID("_EdgeIntensity");
        static readonly int _HatchScaleID     = Shader.PropertyToID("_HatchScale");
        static readonly int _HatchContrastID  = Shader.PropertyToID("_HatchContrast");
        static readonly int _InkColorID       = Shader.PropertyToID("_InkColor");
        static readonly int _PaperColorID     = Shader.PropertyToID("_PaperColor");
        static readonly int _EdgeToonMixID    = Shader.PropertyToID("_EdgeToonMix");
        static readonly int _SceneInfluenceID = Shader.PropertyToID("_SceneInfluence");
        static readonly int _BlitTextureID    = Shader.PropertyToID("_BlitTexture");

        public HatchingPass(string profilerTag, HatchingSettings settings)
        {
            m_ProfilerTag = profilerTag;
            m_Settings = settings;
        }

        // ---- Compatibility Mode path (non-RenderGraph) ----
        [System.Obsolete] // silence URP deprecation warning
        public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
        {
            if (m_Settings.material == null) return;

            var cmd = CommandBufferPool.Get(m_ProfilerTag);
            var mat = m_Settings.material;

            // push uniforms
            mat.SetFloat(_EdgeThicknessID,  m_Settings.edgeThickness);
            mat.SetFloat(_EdgeIntensityID,  m_Settings.edgeIntensity);
            mat.SetFloat(_HatchScaleID,     m_Settings.hatchScale);
            mat.SetFloat(_HatchContrastID,  m_Settings.hatchContrast);
            mat.SetColor(_InkColorID,       m_Settings.inkColor);
            mat.SetColor(_PaperColorID,     m_Settings.paperColor);
            mat.SetFloat(_EdgeToonMixID,    m_Settings.edgeToonMix);
            mat.SetFloat(_SceneInfluenceID, m_Settings.sceneInfluence);

            var renderer = renderingData.cameraData.renderer;
            var src = renderer.cameraColorTargetHandle;

            var desc = renderingData.cameraData.cameraTargetDescriptor;
            desc.msaaSamples = 1;
            desc.depthBufferBits = 0;

            var temp = RTHandles.Alloc(desc, name: "_MoebiusTempColor_Compat");

            Blitter.BlitCameraTexture(cmd, src, temp, mat, 0);
            Blitter.BlitCameraTexture(cmd, temp, src);

            context.ExecuteCommandBuffer(cmd);
            CommandBufferPool.Release(cmd);
            temp?.Release();
        }

        // ---- Render Graph path (URP 17+) ----
        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            if (m_Settings.material == null) return;

            var resources = frameData.Get<UniversalResourceData>();
            TextureHandle srcCamColor = resources.activeColorTexture;

            // Prepare destination (same descriptor as camera color)
            var dstDesc = srcCamColor.GetDescriptor(renderGraph);
            dstDesc.name = "_MoebiusTempColor_RG";
            TextureHandle dst = renderGraph.CreateTexture(dstDesc);

            // Push uniforms BEFORE the blit so the material has correct params
            var mat = m_Settings.material;
            mat.SetFloat(_EdgeThicknessID,  m_Settings.edgeThickness);
            mat.SetFloat(_EdgeIntensityID,  m_Settings.edgeIntensity);
            mat.SetFloat(_HatchScaleID,     m_Settings.hatchScale);
            mat.SetFloat(_HatchContrastID,  m_Settings.hatchContrast);
            mat.SetColor(_InkColorID,       m_Settings.inkColor);
            mat.SetColor(_PaperColorID,     m_Settings.paperColor);
            mat.SetFloat(_EdgeToonMixID,    m_Settings.edgeToonMix);
            mat.SetFloat(_SceneInfluenceID, m_Settings.sceneInfluence);

            // 1) source -> dst using our material (shader pass 0)
            var passParams = new RenderGraphUtils.BlitMaterialParameters(srcCamColor, dst, m_Settings.material, 0);
            renderGraph.AddBlitPass(passParams, "Moebius Hatching");
            resources.cameraColor = dst; // keep this so downstream reads your result

            // If you prefer the classic "copy back", you could instead:
            // var copyParams = new RenderGraphUtils.BlitParameters(dst, srcCamColor);
            // renderGraph.AddBlitPass(copyParams, "Moebius Hatching CopyBack");
        }
    }

    public HatchingSettings settings = new();
    private HatchingPass _pass;

    public override void Create()
    {
        _pass = new HatchingPass("Moebius Hatching", settings)
        {
            renderPassEvent = settings.renderPassEvent
        };
    }

    public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
    {
        if (settings.material == null) return;
        renderer.EnqueuePass(_pass);
    }
}
