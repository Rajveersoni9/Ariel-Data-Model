import React, { useRef, useState, useEffect, Suspense, useCallback } from 'react';
import { Canvas, useFrame, useLoader } from '@react-three/fiber';
import { TextureLoader } from 'three';
import * as THREE from 'three';

const PLANET_TEXTURES = [
  '/exoplanet_magma_texture_1777108562720.png',
  '/exoplanet_thumb_ice_1777108645022.png',
  '/exoplanet_thumb_toxic_1777108674424.png',
  '/exoplanet_thumb_desert_1777108928982.png',
];

const PLANET_NAMES = ['HD-982547 b', 'TOI-1781 c', 'K2-18 b', 'Kepler-1649 c'];

const ExoPlanetMesh = ({ textureUrl }) => {
  const meshRef = useRef();
  const texture = useLoader(TextureLoader, textureUrl);

  useFrame(() => {
    if (meshRef.current) {
      meshRef.current.rotation.y += 0.004;
      meshRef.current.rotation.x += 0.001;
    }
  });

  return (
    <group>
      <mesh ref={meshRef}>
        <sphereGeometry args={[1.5, 64, 64]} />
        <meshStandardMaterial map={texture} roughness={0.6} metalness={0.2} />
      </mesh>
      <mesh>
        <sphereGeometry args={[1.6, 64, 64]} />
        <meshStandardMaterial color="#a855f7" transparent opacity={0.1} side={THREE.FrontSide} depthWrite={false} />
      </mesh>
      <mesh>
        <sphereGeometry args={[1.7, 64, 64]} />
        <meshStandardMaterial color="#7c3aed" transparent opacity={0.05} side={THREE.BackSide} depthWrite={false} />
      </mesh>
    </group>
  );
};

// Fallback if texture hasn't loaded yet
const FallbackPlanet = ({ color = '#a855f7' }) => {
  const meshRef = useRef();
  useFrame(() => { if (meshRef.current) meshRef.current.rotation.y += 0.004; });
  return (
    <mesh ref={meshRef}>
      <sphereGeometry args={[1.5, 32, 32]} />
      <meshStandardMaterial color={color} />
    </mesh>
  );
};

const PlanetCarousel = ({ loading, activeIndex, onIndexChange }) => {
  const [countdown, setCountdown] = useState(2);

  useEffect(() => {
    if (!loading) { setCountdown(2); return; }
    const interval = setInterval(() => {
      setCountdown(prev => {
        if (prev <= 1) {
          onIndexChange(i => (i + 1) % PLANET_TEXTURES.length);
          return 2;
        }
        return prev - 1;
      });
    }, 1000);
    return () => clearInterval(interval);
  }, [loading, onIndexChange]);

  return (
    <>
      {/* Main 3D planet canvas */}
      <div style={{ width: '100%', height: '100%' }}>
        <Canvas camera={{ position: [0, 0, 4], fov: 45 }}>
          <ambientLight intensity={0.2} />
          <pointLight position={[5, 5, 5]} intensity={1.5} color="#ffffff" />
          <pointLight position={[-3, -3, 2]} intensity={0.4} color="#a855f7" />
          <Suspense fallback={<FallbackPlanet />}>
            <ExoPlanetMesh textureUrl={PLANET_TEXTURES[activeIndex]} />
          </Suspense>
        </Canvas>
      </div>

      {/* Right sidebar thumbnails */}
      <div className="right-carousel">
        <div className="carousel-title">Exoplanet<br />Rotation</div>
        {PLANET_TEXTURES.map((tex, i) => (
          <div
            key={i}
            className={`planet-thumb ${i === activeIndex ? 'active' : ''}`}
            onClick={() => onIndexChange(i)}
          >
            <img src={tex} alt={PLANET_NAMES[i]} />
          </div>
        ))}
        {loading && (
          <div className="countdown-box">
            <div className="countdown-label">Next Change In</div>
            <div className="countdown-timer">00:0{countdown}</div>
          </div>
        )}
      </div>
    </>
  );
};

export { PLANET_NAMES };
export default PlanetCarousel;
