Shader "Hidden/MoebiusHatching"
{
    Properties
    {
        _EdgeThickness("Edge Thickness", Float) = 1.0
        _EdgeIntensity("Edge Intensity", Float) = 1.0
        _HatchScale("Hatch Scale", Float) = 1.5
        _HatchContrast("Hatch Contrast", Float) = 0.65
        _InkColor("Ink Color", Color) = (0.05,0.04,0.03,1)
        _PaperColor("Paper Color", Color) = (0.98,0.97,0.92,1)
        _EdgeToonMix("EdgeToon Mix", Range(0,1)) = 0.9
        _SceneInfluence("Scene Influence", Range(0,1)) = 0.25
    }

    SubShader
    {
        Tags { "RenderPipeline"="UniversalPipeline" "RenderType"="Opaque" }
        ZWrite Off ZTest Always Cull Off

        Pass
        {
            Name "MoebiusHatching"
            HLSLPROGRAM
            #pragma target 3.5
            #pragma vertex Vert
            #pragma fragment Frag

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.core/Runtime/Utilities/Blit.hlsl"

            // Camera textures (provided by URP when enabled)
            TEXTURE2D_X_FLOAT(_CameraDepthTexture);   SAMPLER(sampler_CameraDepthTexture);
            TEXTURE2D_X(_CameraNormalsTexture);       SAMPLER(sampler_CameraNormalsTexture);

            // --- Material properties in SRP-batcher friendly cbuffer ---
            CBUFFER_START(UnityPerMaterial)
                float _EdgeThickness;
                float _EdgeIntensity;
                float _HatchScale;
                float _HatchContrast;
                float4 _InkColor;
                float4 _PaperColor;
                float _EdgeToonMix;
                float _SceneInfluence;
            CBUFFER_END

            // --- helpers ---
            float Luma(float3 c) { return dot(c, float3(0.299, 0.587, 0.114)); }

            float LineHatch(float2 p, float angle, float scale, float width)
            {
                float2 d = float2(cos(angle), sin(angle));
                float v = dot(p, d) * scale;
                float f = abs(frac(v) - 0.5) * 2.0; // 0 at line, 1 between
                return smoothstep(0.0, width, 1.0 - f);
            }

            float SobelDepth(float2 uv, float2 texel)
            {
                float d00 = SAMPLE_TEXTURE2D_X(_CameraDepthTexture, sampler_CameraDepthTexture, uv + texel * float2(-1,-1)).r;
                float d10 = SAMPLE_TEXTURE2D_X(_CameraDepthTexture, sampler_CameraDepthTexture, uv + texel * float2( 0,-1)).r;
                float d20 = SAMPLE_TEXTURE2D_X(_CameraDepthTexture, sampler_CameraDepthTexture, uv + texel * float2( 1,-1)).r;
                float d01 = SAMPLE_TEXTURE2D_X(_CameraDepthTexture, sampler_CameraDepthTexture, uv + texel * float2(-1, 0)).r;
                float d21 = SAMPLE_TEXTURE2D_X(_CameraDepthTexture, sampler_CameraDepthTexture, uv + texel * float2( 1, 0)).r;
                float d02 = SAMPLE_TEXTURE2D_X(_CameraDepthTexture, sampler_CameraDepthTexture, uv + texel * float2(-1, 1)).r;
                float d12 = SAMPLE_TEXTURE2D_X(_CameraDepthTexture, sampler_CameraDepthTexture, uv + texel * float2( 0, 1)).r;
                float d22 = SAMPLE_TEXTURE2D_X(_CameraDepthTexture, sampler_CameraDepthTexture, uv + texel * float2( 1, 1)).r;

                float gx = (d20 + 2*d21 + d22) - (d00 + 2*d01 + d02);
                float gy = (d02 + 2*d12 + d22) - (d00 + 2*d10 + d20);
                return saturate(sqrt(gx*gx + gy*gy) * 2.0);
            }

            float SobelNormals(float2 uv, float2 texel)
            {
                float3 n00 = SAMPLE_TEXTURE2D_X(_CameraNormalsTexture, sampler_CameraNormalsTexture, uv + texel * float2(-1,-1)).xyz;
                float3 n10 = SAMPLE_TEXTURE2D_X(_CameraNormalsTexture, sampler_CameraNormalsTexture, uv + texel * float2( 0,-1)).xyz;
                float3 n20 = SAMPLE_TEXTURE2D_X(_CameraNormalsTexture, sampler_CameraNormalsTexture, uv + texel * float2( 1,-1)).xyz;
                float3 n01 = SAMPLE_TEXTURE2D_X(_CameraNormalsTexture, sampler_CameraNormalsTexture, uv + texel * float2(-1, 0)).xyz;
                float3 n21 = SAMPLE_TEXTURE2D_X(_CameraNormalsTexture, sampler_CameraNormalsTexture, uv + texel * float2( 1, 0)).xyz;
                float3 n02 = SAMPLE_TEXTURE2D_X(_CameraNormalsTexture, sampler_CameraNormalsTexture, uv + texel * float2(-1, 1)).xyz;
                float3 n12 = SAMPLE_TEXTURE2D_X(_CameraNormalsTexture, sampler_CameraNormalsTexture, uv + texel * float2( 0, 1)).xyz;
                float3 n22 = SAMPLE_TEXTURE2D_X(_CameraNormalsTexture, sampler_CameraNormalsTexture, uv + texel * float2( 1, 1)).xyz;

                float3 gx = (n20 + 2*n21 + n22) - (n00 + 2*n01 + n02);
                float3 gy = (n02 + 2*n12 + n22) - (n00 + 2*n10 + n20);
                float mag = length(gx) + length(gy);
                return saturate(mag * 0.75);
            }

            // Procedural cross-hatching (three rotated layers)
            float Hatch3(float2 uv, float tone, float2 screenSize)
            {
                float2 p = uv * screenSize / max(screenSize.x, screenSize.y);
                float s = _HatchScale;
                const float width = 0.15;

                float a0 = radians( 15.0);
                float a1 = radians( 75.0);
                float a2 = radians(135.0);

                float l0 = LineHatch(p, a0, s, width);
                float l1 = LineHatch(p, a1, s, width);
                float l2 = LineHatch(p, a2, s, width);

                // Ensure non-negative base for pow; clamp exponent too
                float baseTone = saturate(tone);
                float expVal   = max(1e-3, _HatchContrast * 2.0 + 0.25);
                float t = saturate(pow(baseTone, expVal));

                float h0 = smoothstep(0.9, 0.3, t) * l0;
                float h1 = smoothstep(0.7, 0.2, t) * l1;
                float h2 = smoothstep(0.5, 0.1, t) * l2;

                return saturate(h0 + h1 + h2);
            }

            struct FSOut { float4 pos : SV_POSITION; float2 uv : TEXCOORD0; };

            FSOut Vert(uint id : SV_VertexID)
            {
                FSOut o;
                o.pos = GetFullScreenTriangleVertexPosition(id);
                o.uv  = GetFullScreenTriangleTexCoord(id);
                return o;
            }

            float4 Frag(FSOut i) : SV_Target
            {
                float2 screenSize = float2(_ScreenParams.x, _ScreenParams.y);
                float2 texel = _EdgeThickness / screenSize;

                // Use Blit.hlsl's linear clamp sampler
                float4 src  = SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, i.uv);
                float  tone = 1.0 - Luma(src.rgb);

                float eDepth = SobelDepth(i.uv, texel);
                float eNorm  = SobelNormals(i.uv, texel);
                float edge   = saturate((eDepth * 0.6 + eNorm * 0.8) * _EdgeIntensity);

                float ink = Hatch3(i.uv, tone, screenSize);

                float3 paper   = _PaperColor.rgb;
                float3 inkCol  = _InkColor.rgb;
                float3 hatched = lerp(paper, inkCol, ink);
                float3 edged   = lerp(hatched, inkCol, edge * _EdgeToonMix);

                float3 finalCol = lerp(edged, src.rgb, _SceneInfluence);
                return float4(finalCol, 1);
            }
            ENDHLSL
        }
    }
    Fallback Off
}
