"""Ревизия п. 1.11 (Жорик №7 от 23.10.2026): поза r и r′ по детекциям YuNet, без распознавания; R′, r0′ и доли снимков, у которых r и r′
по разные стороны r0 и R. Детекции – detections.tsv данных и derived/detections.tsv клона FERET.

python3 -I rev_pose.py
"""
import collections, sys
def r(row): return (row[8]-(row[4]+row[6])/2)/abs(row[6]-row[4])
def rp(row):
    ex,ey=row[6]-row[4],row[7]-row[5]
    if ex<0: ex,ey=-ex,-ey
    mx,my=(row[4]+row[6])/2,(row[5]+row[7])/2
    return ((row[8]-mx)*ex+(row[9]-my)*ey)/(ex*ex+ey*ey)
def pct(v,p): s=sorted(v); return s[int(round(p*(len(s)-1)))]
rows=collections.defaultdict(dict)
for l in open("/home/user/svd-faces-data/detections.tsv",encoding="utf-8"):
    c=l.rstrip("\n").split("\t")
    if c[0]=="base" or c[4]!="1": continue
    rows[c[0]][c[1]+"/"+c[2]]=[float(x) for x in c[5:20]]
for l in open("/home/user/svd-feret-cloud/derived/detections.tsv",encoding="utf-8"):
    c=l.rstrip("\n").split("\t")
    if len(c)>=17 and c[1]=="1": rows["feret"][c[0]]=[float(x) for x in c[2:17]]
inc=set()
for l in open("/home/user/svd-faces-data/fei_selection.tsv",encoding="utf-8"):
    c=l.rstrip("\n").split("\t")
    if l.startswith("#") or c[0]=="person": continue
    if c[6]=="1" and 11<=int(c[1])<=14: inc.add(c[0]+"/"+c[2])
R=pct([abs(r(x)) for x in rows["own"].values()],0.95); Rp=pct([abs(rp(x)) for x in rows["own"].values()],0.95)
fa=[v for k,v in rows["fei"].items() if k in inc]
r0=pct([abs(r(x)) for x in fa],0.95); r0p=pct([abs(rp(x)) for x in fa],0.95)
print("own n=%d R=%.6f R'=%.6f; FEI №11–14 n=%d r0=%.6f r0'=%.6f (решение r0=0,12)"%(len(rows["own"]),R,Rp,len(fa),r0,r0p))
R0=0.12
print("база | снимков | (i) пороги прежние: r0 / R / любой | (ii) r′ с r0′, R′: r0 / R / любой | роль |roll| медиана, 95 %")
import math
for b in ("gt","own","muct","fei","feret","orl"):
    v=list(rows[b].values()); n=len(v)
    def cnt(a0,aR):
        x0=sum((abs(r(x))<=R0)!=(abs(rp(x))<=a0) for x in v); xR=sum((abs(r(x))<=R)!=(abs(rp(x))<=aR) for x in v)
        xa=sum(((abs(r(x))<=R0)!=(abs(rp(x))<=a0)) or ((abs(r(x))<=R)!=(abs(rp(x))<=aR)) for x in v)
        return "%.1f / %.1f / %.1f %%"%(100*x0/n,100*xR/n,100*xa/n)
    rolls=sorted(abs(math.degrees(math.atan2(x[7]-x[5],x[6]-x[4]))) for x in v)
    print(b,n,cnt(R0,R),"|",cnt(r0p,Rp),"| roll %.1f°, %.1f°"%(rolls[n//2],rolls[int(0.95*(n-1))]))
