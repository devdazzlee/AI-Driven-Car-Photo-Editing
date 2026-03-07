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
  const isTouchDragging = useRef(false);

  const updatePosition = useCallback((clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = clientX - rect.left;
    const pct = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setPosition(pct);
  }, []);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (isDragging.current) {
        e.preventDefault();
        updatePosition(e.clientX);
      }
    };
    const handleMouseUp = () => {
      isDragging.current = false;
    };
    window.addEventListener("mousemove", handleMouseMove, { passive: false });
    window.addEventListener("mouseup", handleMouseUp);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, [updatePosition]);

  useEffect(() => {
    const handleTouchMove = (e: TouchEvent) => {
      if (isTouchDragging.current && e.touches.length > 0) {
        e.preventDefault();
        updatePosition(e.touches[0].clientX);
      }
    };
    const handleTouchEnd = () => {
      isTouchDragging.current = false;
    };
    window.addEventListener("touchmove", handleTouchMove, { passive: false });
    window.addEventListener("touchend", handleTouchEnd);
    window.addEventListener("touchcancel", handleTouchEnd);
    return () => {
      window.removeEventListener("touchmove", handleTouchMove);
      window.removeEventListener("touchend", handleTouchEnd);
      window.removeEventListener("touchcancel", handleTouchEnd);
    };
  }, [updatePosition]);

  const handlePointerDown = useCallback(
    (e: React.MouseEvent | React.TouchEvent) => {
      e.preventDefault();
      if ("touches" in e) {
        isTouchDragging.current = true;
        if (e.touches.length > 0) updatePosition(e.touches[0].clientX);
      } else {
        isDragging.current = true;
        updatePosition((e as React.MouseEvent).clientX);
      }
    },
    [updatePosition]
  );

  const bothLoaded = beforeLoaded && afterLoaded;

  const handleContainerPointerDown = useCallback(
    (e: React.MouseEvent | React.TouchEvent) => {
      if (!bothLoaded) return;
      const clientX = "touches" in e ? e.touches[0]?.clientX : (e as React.MouseEvent).clientX;
      if (typeof clientX === "number") updatePosition(clientX);
    },
    [bothLoaded, updatePosition]
  );

  return (
    <div className="overflow-hidden rounded-2xl border border-slate-200 bg-slate-100 shadow-sm dark:border-slate-700 dark:bg-slate-800">
      <div
        ref={containerRef}
        className="relative aspect-[4/3] cursor-col-resize select-none sm:aspect-[16/10]"
        style={{ touchAction: "none" }}
        onMouseDown={handleContainerPointerDown}
        onTouchStart={handleContainerPointerDown}
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
          draggable={false}
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
            draggable={false}
            className={cn(
              "h-full w-full object-contain transition-opacity duration-300",
              afterLoaded ? "opacity-100" : "opacity-0"
            )}
            onLoad={() => setAfterLoaded(true)}
          />
        </div>
        <div
          className="absolute top-0 bottom-0 z-20 flex w-10 -translate-x-1/2 cursor-grab items-center justify-center active:cursor-grabbing"
          style={{ left: `${position}%`, pointerEvents: bothLoaded ? "auto" : "none" }}
          onMouseDown={handlePointerDown}
          onTouchStart={handlePointerDown}
        >
          <div className="flex h-12 w-6 shrink-0 items-center justify-center rounded-full bg-white shadow-lg ring-2 ring-slate-200 transition active:scale-95 active:ring-emerald-300 dark:ring-slate-600">
            <GripVertical className="h-5 w-5 text-slate-600" strokeWidth={2.5} />
          </div>
          <div className="absolute inset-y-0 left-1/2 w-0.5 -translate-x-1/2 bg-white shadow-lg" aria-hidden />
        </div>
        <div className="absolute left-3 top-3 rounded-lg bg-slate-900 px-2.5 py-1.5 text-xs font-medium text-white">
          {beforeLabel} / {afterLabel} • drag to compare
        </div>
      </div>
    </div>
  );
}
