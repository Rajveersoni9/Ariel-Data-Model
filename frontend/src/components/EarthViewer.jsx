import React, { useRef, Suspense } from 'react';
import { Canvas, useFrame, useLoader } from '@react-three/fiber';
import { TextureLoader } from 'three';
import { Stars } from '@react-three/drei';
import * as THREE from 'three';

const EarthMesh = () => {
  const meshRef = useRef();
  const atmosphereRef = useRef();
  const texture = useLoader(TextureLoader, '/earth_night_texture_1777108542229.png');

  useFrame(() => {
    if (meshRef.current) meshRef.current.rotation.y += 0.003;
    if (atmosphereRef.current) atmosphereRef.current.rotation.y += 0.002;
  });

  return (
    <group>
      <mesh ref={meshRef}>
        <sphereGeometry args={[1.5, 64, 64]} />
        <meshStandardMaterial map={texture} roughness={0.8} metalness={0.1} />
      </mesh>
      <mesh ref={atmosphereRef}>
        <sphereGeometry args={[1.58, 64, 64]} />
        <meshStandardMaterial color="#1a6aff" transparent opacity={0.12} side={THREE.FrontSide} depthWrite={false} />
      </mesh>
      <mesh>
        <sphereGeometry args={[1.65, 64, 64]} />
        <meshStandardMaterial color="#3399ff" transparent opacity={0.05} side={THREE.BackSide} depthWrite={false} />
      </mesh>
    </group>
  );
};

// Fallback sphere if texture fails to load
const FallbackEarth = () => {
  const meshRef = useRef();
  useFrame(() => { if (meshRef.current) meshRef.current.rotation.y += 0.003; });
  return (
    <mesh ref={meshRef}>
      <sphereGeometry args={[1.5, 64, 64]} />
      <meshStandardMaterial color="#1a4aff" wireframe />
    </mesh>
  );
};

const EarthViewer = () => {
  return (
    <div style={{ width: '100%', height: '100%' }}>
      <Canvas camera={{ position: [0, 0, 4], fov: 45 }}>
        <ambientLight intensity={0.15} />
        <directionalLight position={[5, 3, 5]} intensity={1.2} color="#ffffff" />
        <pointLight position={[-5, -3, -3]} intensity={0.2} color="#1a4aff" />
        <Stars radius={100} depth={50} count={3000} factor={3} fade speed={0.5} />
        <Suspense fallback={<FallbackEarth />}>
          <EarthMesh />
        </Suspense>
      </Canvas>
    </div>
  );
};

export default EarthViewer;
