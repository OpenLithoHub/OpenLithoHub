#!/usr/bin/env python3
"""Replay B04 Increment 15 halo-tail upper/lower certificates."""
from __future__ import annotations

from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
import json
import math
from pathlib import Path

import numpy as np

SNAPSHOT = Path("B04_K24_HaloKernelSnapshot_2026-09-09.npz")
OUT = Path("B04_Increment15_HaloTail_replayed.json")

z = np.load(SNAPSHOT, allow_pickle=False)
kern = z["kernels_spatial"].astype(np.complex64)
weights = z["weights24"].astype(np.float32)
N = int(z["grid_n"])
pixel_nm = float(z["pixel_nm"])
K = kern.shape[0]
center = N // 2
yy, xx = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
dist = np.maximum(np.abs(yy-center), np.abs(xx-center))
max_h = int(dist.max())

tail_lo = np.zeros((K,max_h+1), dtype=np.float64)
tail_hi = np.zeros((K,max_h+1), dtype=np.float64)
a_hi = np.zeros(K, dtype=np.float64)

for rounding, target in ((ROUND_FLOOR,tail_lo),(ROUND_CEILING,tail_hi)):
    with localcontext() as ctx:
        ctx.prec=100
        ctx.rounding=rounding
        for j in range(K):
            rings=[]
            for r in range(max_h+1):
                s=Decimal(0)
                for y,x in np.argwhere(dist==r):
                    c=kern[j,y,x]
                    re=Decimal.from_float(float(np.float32(c.real)))
                    im=Decimal.from_float(float(np.float32(c.imag)))
                    s += (re*re+im*im).sqrt(context=ctx)
                rings.append(s)
            if rounding == ROUND_CEILING:
                a_hi[j]=np.nextafter(float(sum(rings,Decimal(0))),np.inf)
            for h in range(max_h+1):
                if h == max_h:
                    target[j,h]=0.0
                else:
                    s=sum(rings[h+1:],Decimal(0))
                    target[j,h]=np.nextafter(
                        float(s), -np.inf if rounding == ROUND_FLOOR else np.inf
                    )

w_hi=np.nextafter(weights.astype(np.float64),np.inf)
w_lo=np.nextafter(weights.astype(np.float64),-np.inf)
pi_hi=np.nextafter(math.pi,np.inf)
profile=[]
for h in range(max_h+1):
    t=tail_hi[:,h]
    upper=np.nextafter(np.sum(w_hi*(2*a_hi*t+t*t)),np.inf)
    lower=np.nextafter(np.max(w_lo*(tail_lo[:,h]/pi_hi)**2),-np.inf)
    profile.append({
        "h_px":h,
        "h_nm":h*pixel_nm,
        "unrestricted_binary_lower_intensity":max(0.0,float(lower)),
        "absolute_tail_upper_intensity":float(upper),
    })

def bracket(eps):
    bad=[r["h_px"] for r in profile[:-1]
         if r["unrestricted_binary_lower_intensity"] > eps]
    necessary=max(bad)+1 if bad else 0
    sufficient=next(
        (r["h_px"] for r in profile
         if r["absolute_tail_upper_intensity"] <= eps),
        None,
    )
    return {"epsilon_intensity":eps,
            "necessary_h_min_px":necessary,
            "certified_sufficient_h_px":sufficient}

result={
    "status":"PASS_REPLAY",
    "profile":profile,
    "threshold_brackets":[
        bracket(0.005752571687594228),
        bracket(0.002916392),
        bracket(0.001),
    ],
}
OUT.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result["threshold_brackets"],indent=2))
