import React, { useEffect, useRef, forwardRef } from 'react';
import '../styles/AudioVisualizer.css';

export const AudioVisualizer = forwardRef(({ audioLevel = 0, isActive = false }, ref) => {
  const canvasRef = useRef(null);
  const animationFrameId = useRef(null);
  const smoothLevel = useRef(0);
  const bars = useRef(Array(32).fill(0));
  
  // Set up the canvas and animation
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Set canvas dimensions
    const updateCanvasSize = () => {
      const container = canvas.parentElement;
      if (container) {
        const size = Math.min(container.clientWidth, container.clientHeight) * 0.8;
        canvas.width = size;
        canvas.height = size;
      }
    };
    
    updateCanvasSize();
    window.addEventListener('resize', updateCanvasSize);
    
    return () => {
      window.removeEventListener('resize', updateCanvasSize);
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current);
      }
    };
  }, []);
  
  // Animation loop
  useEffect(() => {
    if (!isActive) return;
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const draw = () => {
      // Smooth the audio level
      smoothLevel.current += (audioLevel - smoothLevel.current) * 0.2;
      
      // Update bars with new audio level
      bars.current = bars.current.map((bar, i) => {
        // Add some randomness to make it more organic
        const target = smoothLevel.current * (0.8 + Math.random() * 0.4);
        return bar + (target - bar) * 0.1;
      });
      
      // Draw the visualization
      const { width, height } = canvas;
      const centerX = width / 2;
      const centerY = height / 2;
      const maxRadius = Math.min(width, height) * 0.4;
      const barWidth = 4;
      const barCount = 32;
      
      // Clear canvas
      ctx.clearRect(0, 0, width, height);
      
      // Draw bars in a circle
      for (let i = 0; i < barCount; i++) {
        const angle = (i / barCount) * Math.PI * 2;
        const barHeight = bars.current[i] * maxRadius * 0.8 + maxRadius * 0.2;
        
        ctx.save();
        ctx.translate(centerX, centerY);
        ctx.rotate(angle);
        
        // Draw bar
        ctx.fillStyle = `hsl(${i * 10}, 80%, 60%)`;
        ctx.fillRect(
          -barWidth / 2,
          -maxRadius,
          barWidth,
          -barHeight
        );
        
        // Draw reflection
        const gradient = ctx.createLinearGradient(0, -maxRadius, 0, -barHeight);
        gradient.addColorStop(0, 'rgba(255, 255, 255, 0.5)');
        gradient.addColorStop(1, 'rgba(255, 255, 255, 0)');
        ctx.fillStyle = gradient;
        ctx.fillRect(
          -barWidth / 2,
          -maxRadius,
          barWidth,
          -barHeight
        );
        
        ctx.restore();
      }
      
      // Continue the animation
      animationFrameId.current = requestAnimationFrame(draw);
    };
    
    // Start the animation
    animationFrameId.current = requestAnimationFrame(draw);
    
    return () => {
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current);
      }
    };
  }, [isActive, audioLevel]);
  
  return (
    <div className="audio-visualizer">
      <canvas 
        ref={canvasRef} 
        className={`visualizer-canvas ${isActive ? 'active' : ''}`}
      />
    </div>
  );
});

AudioVisualizer.displayName = 'AudioVisualizer';
