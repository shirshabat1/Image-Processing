Shader "CG/Earth"
{
    Properties
    {
        [NoScaleOffset] _AlbedoMap ("Albedo Map", 2D) = "defaulttexture" {}
        _Ambient ("Ambient", Range(0, 1)) = 0.15
        [NoScaleOffset] _SpecularMap ("Specular Map", 2D) = "defaulttexture" {}
        _Shininess ("Shininess", Range(0.1, 100)) = 50
        [NoScaleOffset] _HeightMap ("Height Map", 2D) = "defaulttexture" {}
        _BumpScale ("Bump Scale", Range(1, 100)) = 30
        [NoScaleOffset] _CloudMap ("Cloud Map", 2D) = "black" {}
        _AtmosphereColor ("Atmosphere Color", Color) = (0.8, 0.85, 1, 1)
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
                uniform sampler2D _CloudMap;
                uniform fixed4 _AtmosphereColor;

                struct appdata
                { 
                    float4 vertex : POSITION;
                };

                struct v2f
                {
                    float4 pos : SV_POSITION;
                    float4 worldPos : TEXTCOORD1;
                    float3 vertex : NORMAL;
                };

                v2f vert (appdata input)    
                {
                    v2f output;
                    output.pos = UnityObjectToClipPos(input.vertex);
                    output.worldPos = mul(unity_ObjectToWorld, input.vertex);
                    output.vertex = input.vertex;
                    return output;
                }

                fixed4 frag (v2f input) : SV_Target
                {
                    float2 uv = getSphericalUV(input.worldPos);

                    float3 l = normalize(_WorldSpaceLightPos0);
                    float3 v = normalize(_WorldSpaceCameraPos - input.worldPos);
                    float3 h = normalize((l+v) / 2);

                    float3 n = normalize(mul(unity_ObjectToWorld, input.vertex));

                    bumpMapData i; 
                    i.normal = normalize(n);
                    i.tangent = cross(n, float3(0,1,0));
                    i.uv = uv; 
                    i.heightMap = _HeightMap;
                    i.du = _HeightMap_TexelSize.x;
                    i.dv = _HeightMap_TexelSize.y;
                    i.bumpScale = _BumpScale / 10000;

                    fixed lambert = max(0, dot(n,l)); 
                    fixed3 view_dir = normalize(_WorldSpaceCameraPos);
                    float atmosphere = (1-max(0, dot(n,view_dir)))*sqrt(lambert)*_AtmosphereColor;
                    float clouds =  tex2D(_CloudMap, uv)  * (sqrt(lambert) + _Ambient);

                    float3 bump_n = normalize(getBumpMappedNormal(i));
                    fixed3 finalNormal = (1-tex2D(_SpecularMap, uv)) * bump_n + tex2D(_SpecularMap, uv)*n ;
                    fixed3 blinnPhongRes = blinnPhong(finalNormal, v, l, _Shininess, tex2D(_AlbedoMap, uv), tex2D(_SpecularMap, uv), _Ambient);

                    fixed4 res = float4(blinnPhongRes + atmosphere + clouds,0);
                    return res;
                }

            ENDCG
        }
    }
}   
