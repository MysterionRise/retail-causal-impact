// path: web/lib/useApi.ts
"use client";

import { useEffect, useRef, useState } from "react";

/**
 * Minimal data hook: runs `loader` whenever `deps` change, keeps the last
 * good value while reloading, debounces rapid changes (sliders) and ignores
 * out-of-order responses.
 */
export function useApi<T>(loader: () => Promise<T>, deps: unknown[], debounceMs = 0) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const seq = useRef(0);

  useEffect(() => {
    const mySeq = ++seq.current;
    const timer = setTimeout(() => {
      setLoading(true);
      loader()
        .then((d) => {
          if (seq.current !== mySeq) return;
          setData(d);
          setError(null);
        })
        .catch((e: unknown) => {
          if (seq.current !== mySeq) return;
          setError(e instanceof Error ? e.message : String(e));
        })
        .finally(() => {
          if (seq.current === mySeq) setLoading(false);
        });
    }, debounceMs);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return { data, error, loading };
}
