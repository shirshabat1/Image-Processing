Shader "CG/Bricks"
{
    Properties
    {
        [NoScaleOffset] _AlbedoMap ("Albedo Map", 2D) = "defaulttexture" {}
        _Ambient ("Ambient", Range(0, 1)) = 0.15
        [NoScaleOffset] _SpecularMap ("Specular Map", 2D) = "defaulttexture" {}
        _Shininess ("Shininess", Range(0.1, 100)) = 50
        [NoScaleOffset] _HeightMap ("Height Map", 2D) = "defaulttexture" {}
        _BumpScale ("Bump Scale", Range(-100, 100)) = 40
    }
    SubShader
    {
        Pass
        {
            Tags { "LightMode" = "ForwardBase" }

            CGPROGRAM

                #pragma vertex vert
                #pragma fragment frag
                #include "UnityCG.cginc"
                #include "CGUtils.cginc"

                // Declare used properties
                uniform sampler2D _AlbedoMap;
                uniform float _Ambient;
                uniform sampler2D _SpecularMap;
                uniform float _Shininess;
                uniform sampler2D _HeightMap;
                uniform float4 _HeightMap_TexelSize;
                uniform float _BumpScale;

                struct appdata
                { 
                    float4 vertex   : POSITION;
                    float3 normal   : NORMAL;
                    float4 tangent  : TANGENT;
                    float2 uv       : TEXCOORD0;
                };

                struct v2f
                {
                    float4 pos : SV_POSITION;
                    float2 uv  : TEXCOORD0;
                    float4 tangent  : TANGENT;
                    float3 normal : NORMAL;
                    float4 worldPos : TEXTCOORD1;
                };



                v2f vert (appdata input)
                {
                    v2f output;
                    output.worldPos = mul(unity_ObjectToWorld, input.vertex);
                    output.pos = UnityObjectToClipPos(input.vertex);
                    output.uv = input.uv;
                    output.tangent = mul(unity_ObjectToWorld,input.tangent);
                    output.normal = mul(unity_ObjectToWorld, input.normal);
                    return output;
                }
                fixed4 frag (v2f input) : SV_Target
                {
                    float3 l = normalize(_WorldSpaceLightPos0);
                    float3 v = normalize(_WorldSpaceCameraPos - input.worldPos);
                    float3 h = normalize((l+v) / 2);
                    
                    bumpMapData i; 
                    i.normal = normalize(input.normal);
                    i.tangent = input.tangent;
                    i.uv = input.uv; 
                    i.heightMap = _HeightMap;
                    i.du = _HeightMap_TexelSize.x;
                    i.dv = _HeightMap_TexelSize.y;
                    i.bumpScale = _BumpScale / 10000;
                    float3 n = normalize(getBumpMappedNormal(i));
                    fixed3 blinnPhongRes = blinnPhong(n, v, l, _Shininess, tex2D(_AlbedoMap, input.uv), tex2D(_SpecularMap, input.uv), _Ambient);
                    fixed4 res = float4(blinnPhongRes,0);
                    return res;
                }

            ENDCG
        }
    }
}
