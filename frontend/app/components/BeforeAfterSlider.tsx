"use client";

import { useRef, useState, useCallback, useEffect } from "react";
import { GripVertical, Loader2 } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";

type Props = {
  beforeSrc: string;
  afterSrc: string;
  beforeLabel?: string;
  afterLabel?: string;
};

export function BeforeAfterSlider({
  beforeSrc,
  afterSrc,
  beforeLabel = "Before",
  afterLabel = "After",
}: Props) {
  const [position, setPosition] = useState(50);
  const [beforeLoaded, setBeforeLoaded] = useState(false);
  const [afterLoaded, setAfterLoaded] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const isDragging = useRef(false);

  const updatePosition = useCallback(
    (clientX: number) => {
      if (!containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const x = clientX - rect.left;
      const pct = Math.max(0, Math.min(100, (x / rect.width) * 100));
      setPosition(pct);
    },
    []
  );

  const handleMove = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      e.preventDefault();
      updatePosition(e.clientX);
    },
    [updatePosition]
  );

  const handleTouchMove = useCallback(
    (e: React.TouchEvent<HTMLDivElement>) => {
      if (e.touches.length > 0) updatePosition(e.touches[0].clientX);
    },
    [updatePosition]
  );

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (isDragging.current) updatePosition(e.clientX);
    };
    const handleMouseUp = () => {
      isDragging.current = false;
    };
    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, [updatePosition]);

  const handleMouseDown = () => {
    isDragging.current = true;
  };

  const bothLoaded = beforeLoaded && afterLoaded;

  return (
    <div className="overflow-hidden rounded-2xl border border-slate-200/80 bg-slate-100 shadow-sm dark:border-slate-700/80 dark:bg-slate-800/50">
      <div
        ref={containerRef}
        className="relative aspect-[4/3] cursor-col-resize select-none touch-none sm:aspect-[16/10]"
        onMouseMove={handleMove}
        onMouseLeave={() => !isDragging.current && setPosition(50)}
        onTouchMove={handleTouchMove}
        onTouchEnd={() => setPosition(50)}
      >
        {!bothLoaded && (
          <div className="absolute inset-0 z-10 flex items-center justify-center">
            <Skeleton className="absolute inset-0 rounded-none" />
            <div className="relative z-20 flex flex-col items-center gap-3">
              <Loader2 className="h-8 w-8 animate-spin text-emerald-600" />
              <span className="text-sm font-medium text-slate-600 dark:text-slate-400">
                Loading images…
              </span>
            </div>
          </div>
        )}

        <img
          src={beforeSrc}
          alt={beforeLabel}
          className={cn(
            "absolute inset-0 h-full w-full object-contain transition-opacity duration-300",
            beforeLoaded ? "opacity-100" : "opacity-0"
          )}
          onLoad={() => setBeforeLoaded(true)}
        />
        <div
          className="absolute inset-0 overflow-hidden"
          style={{ clipPath: `inset(0 ${100 - position}% 0 0)` }}
        >
          <img
            src={afterSrc}
            alt={afterLabel}
            className={cn(
              "h-full w-full object-contain transition-opacity duration-300",
              afterLoaded ? "opacity-100" : "opacity-0"
            )}
            onLoad={() => setAfterLoaded(true)}
          />
        </div>
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-white shadow-lg"
          style={{ left: `${position}%`, transform: "translateX(-50%)" }}
        >
          <div
            className="absolute left-1/2 top-1/2 flex -translate-x-1/2 -translate-y-1/2 cursor-grab items-center justify-center rounded-full bg-white p-2 shadow-lg transition active:cursor-grabbing active:scale-95 disabled:pointer-events-none disabled:opacity-50"
            onMouseDown={handleMouseDown}
            onTouchStart={handleMouseDown}
            style={{ pointerEvents: bothLoaded ? "auto" : "none" }}
          >
            <GripVertical className="h-5 w-5 text-slate-600" strokeWidth={2.5} />
          </div>
        </div>
        <div className="absolute left-3 top-3 rounded-lg bg-black/60 px-2.5 py-1.5 text-xs font-medium text-white backdrop-blur-sm">
          {beforeLabel} / {afterLabel} • drag to compare
        </div>
      </div>
    </div>
  );
}
