#!/usr/bin/env wolframscript
(* ::Package:: *)

(* ::Title:: *)
(*HiCAT Aperture Mask Generation*)


(* ::Subtitle:: *)
(*Initialization*)


(* ::Section:: *)
(*Directory*)


(* ::Subsection:: *)
(*FilePath definition*)


(*HiCATDirectory=ParentDirectory[NotebookDirectory[]]<>"/input_files/"*)
HiCATDirectory="/user/kstlaurent/SCDA_2/input_files/"


(* ::Subsection:: *)
(*Padding For FileName function*)


PaddingForFileNames[x_,ndigit_]:=PaddedForm[x,ndigit,NumberPadding->{"0",""},NumberSigns->{"",""}]//ToString
PaddingForFileNamesWithSign[x_,ndigit_]:=PaddedForm[x,ndigit,NumberPadding->{"0",""},NumberSigns->{"-","+"}]//ToString


(* ::Section:: *)
(*Definitions [autoinit]*)


(* ::Subsection:: *)
(*Initializations*)


PackageVersion="5.0.9";
Init[PackageVersion]


Needs["Developer`"]


Off[SFT::"NewPlan"]
Off[ToLyotPlane::"NewPlan"]
SetOptions[ListLogPlot,Joined->True,Frame->True,Axes->None,PlotStyle->AbsoluteThickness[1.5]];


(* ::Section::Closed:: *)
(*Functions*)


(* ::Subsection::Closed:: *)
(*ToLyotPlane center*)


(* ::Subsubsection::Closed:: *)
(*Plane to Lyot plane Center*)


CurrentExistingPlan={};
Options[PlanToLyotPlaneCenter]={MaskSampling->100,Wavelength->1,OverSizeLyot->1,OverSizePupil->1,SoftMask->False};
PlanToLyotPlaneCenter[Pupil_,Mask\[Lambda]surD_,Opt:OptionsPattern[]]:=Module[{n,nMask,Foyer,DiffractedWave,\[Lambda],CoeffOverSize,CoeffOverSizePup,centeringstyle},
n=Dimensions[Pupil][[1]];
nMask= OptionValue[MaskSampling];
\[Lambda]= OptionValue[Wavelength];
CoeffOverSize=OptionValue[OverSizeLyot];
CoeffOverSizePup=OptionValue[OverSizePupil];
TestSoft= OptionValue[SoftMask];
OptsSFT=Sequence@@FilterRules[Flatten[{Opt}],Options[SFT]];
nMask= OptionValue[MaskSampling];
CurrentExistingPlan={n,Mask\[Lambda]surD,nMask,\[Lambda],CoeffOverSize,CoeffOverSizePup,TestSoft};If[TestSoft==True,Mask=Developer`ToPackedArray[Cr\[EAcute]eDiaphragmeLisseCenter[nMask/2,nMask+10,5]];PlanSFT[Pupil,(CoeffOverSizePup Mask\[Lambda]surD (nMask+10))/(\[Lambda] nMask),nMask+10, Centering -> "NoPixel"];PlaniSFT[Mask,1/(nMask \[Lambda]) (nMask+10) CoeffOverSizePup CoeffOverSize Mask\[Lambda]surD,CoeffOverSize n, Centering -> "NoPixel"],Mask=Developer`ToPackedArray[Cr\[EAcute]eDiaphragmeCenter[nMask/2,nMask]];
PlanSFT[Pupil,(CoeffOverSizePup Mask\[Lambda]surD)/\[Lambda],nMask, Centering -> "NoPixel"];PlaniSFT[Mask,(CoeffOverSizePup CoeffOverSize Mask\[Lambda]surD)/\[Lambda],CoeffOverSize n, Centering -> "NoPixel"];]]



(* ::Subsubsection::Closed:: *)
(*To Lyot Plane center*)


Protect[MaskSampling,Wavelength,OverSizeLyot,OverSizePupil];
Options[ToLyotPlaneCenter]={MaskSampling->100,Wavelength->1,OverSizeLyot->1,OverSizePupil->1,SoftMask->False};
ToLyotPlaneCenter::"NewPlan"="Calculating new SFT and iSFT plans...";
ToLyotPlaneCenter[Pupil_,Mask\[Lambda]surD_,Opt:OptionsPattern[]]:=Module[{n,nMask,Foyer,DiffractedWave,\[Lambda],CoeffOverSize,CoeffOverSizePup,centeringstyle},
n=Dimensions[Pupil][[1]];
nMask=OptionValue[MaskSampling];
\[Lambda]=OptionValue[Wavelength];
CoeffOverSize= OptionValue[OverSizeLyot];
CoeffOverSizePup=OptionValue[OverSizePupil];
TestSoft=OptionValue[SoftMask];
If[CurrentExistingPlan!={n,Mask\[Lambda]surD,nMask,\[Lambda],CoeffOverSize,CoeffOverSizePup,TestSoft},Message[ToLyotPlaneCenter::"NewPlan"];NewOpts={MaskSampling->nMask,Wavelength->\[Lambda],OverSizeLyot->CoeffOverSize,OverSizePupil->CoeffOverSizePup,SoftMask->TestSoft};PlanToLyotPlaneCenter[Pupil,Mask\[Lambda]surD,NewOpts]];
If[TestSoft==True,
Foyer=SFT[Pupil,(CoeffOverSizePup Mask\[Lambda]surD (nMask+10))/(\[Lambda] nMask),nMask+10, Centering -> "NoPixel"];DiffractedWave=iSFT[Mask Foyer,1/(nMask \[Lambda]) (nMask+10) CoeffOverSizePup CoeffOverSize Mask\[Lambda]surD,CoeffOverSize n, Centering -> "NoPixel"];DiffractedWave=Resize[Pupil,CoeffOverSize n]-DiffractedWave,
Foyer=Developer`ToPackedArray[SFT[Pupil,(CoeffOverSizePup Mask\[Lambda]surD)/\[Lambda],nMask, Centering -> "NoPixel"]];DiffractedWave=Developer`ToPackedArray[iSFT[Mask Foyer,(CoeffOverSizePup CoeffOverSize Mask\[Lambda]surD)/\[Lambda],CoeffOverSize n, Centering -> "NoPixel"]];DiffractedWave=Developer`ToPackedArray[Resize[Pupil,CoeffOverSize n]-DiffractedWave]]];



(* ::Subsection:: *)
(*Segment generation*)


(* ::Subsubsection::Closed:: *)
(*Segment structure*)


SegStructureFunction[nRing_, Radius_, GapHexagonFraction_, MissingSegmentVector_]:=Module[{nSeg, RingTable, MissingTable, CxTable, CyTable,BordTable,sTable, Rad,fact, i,b, nSegRing,start,refx, refy,iRing, xx, yy, iSeg, cx, cy, \[Rho]Table, PATable, aa, bb, cc, dd, ee, ff},

nSeg=3*nRing*(nRing+1)+1;
RingTable=Table[0, {nSeg}];
MissingTable=Table[0,{nSeg}];
CxTable=Table[0., {nSeg}];
CyTable=Table[0., {nSeg}];
BordTable=Table[{-1, -1, -1, -1, -1, -1}, {nSeg}];
sTable=Table[{-1, -1, -1, -1, -1, -1}, {nSeg}];
Rad=Radius*(1+GapHexagonFraction);
fact=Sqrt[3.];

sTable[[1]]=Range[6];
BordTable[[1]]=Range[6]-1;

If[Length[MissingSegmentVector] >0, 
Do[
If[MissingSegmentVector[[iMissing]]<=  nSeg-1 && MissingSegmentVector[[iMissing]]>=0,
MissingTable[[MissingSegmentVector[[iMissing]]+1]]=1];
, {iMissing, 1, Length[MissingSegmentVector]}]];

i=2;
b=6;
Do[

nSegRing=6*iRing;

start=i;

(*Top Segment*)
refx=0.;
refy=iRing*Rad*fact;
RingTable[[i]]=iRing;
CxTable[[i]]=refx;
CyTable[[i]]=refy;
BordTable[[i,3]]=b++;
i++;

(*Going South-East*)
refx=CxTable[[i-1]];
refy=CyTable[[i-1]];
Do[
xx=refx +s*3*Rad/2.;
yy=refy - s*Rad*fact/2.;
RingTable[[i]]=iRing;
CxTable[[i]]=xx;
CyTable[[i]]=yy;
If[s < iRing,BordTable[[i,3]]=b++,BordTable[[i,4]]=b++];
i++;
,{s, 1, iRing}];


(*going South*)
refx=CxTable[[i-1]];
refy=CyTable[[i-1]];
Do[
xx=refx + 0;
yy=refy - s*Rad*fact;
RingTable[[i]]=iRing;
CxTable[[i]]=xx;
CyTable[[i]]=yy;
If[s < iRing,BordTable[[i,4]]=b++,BordTable[[i,5]]=b++];
i++;
, {s, 1, iRing}];


(*going South-West*)
refx=CxTable[[i-1]];
refy=CyTable[[i-1]];
Do[
xx=refx - s*3.*Rad/2.;
yy=refy - s*Rad*fact/2.;
RingTable[[i]]=iRing;
CxTable[[i]]=xx;
CyTable[[i]]=yy;
If[s < iRing,BordTable[[i,5]]=b++,BordTable[[i,6]]=b++];
i++;
, {s, 1, iRing}];


(*going North-West*)
refx=CxTable[[i-1]];
refy=CyTable[[i-1]];
Do[
xx=refx -s*3.*Rad/2.;
yy=refy + s*Rad*fact/2.;
RingTable[[i]]=iRing;
CxTable[[i]]=xx;
CyTable[[i]]=yy;
If[s < iRing,BordTable[[i,6]]=b++,BordTable[[i,1]]=b++];
i++;
, {s, 1, iRing}];


(*going North*)
refx=CxTable[[i-1]];
refy=CyTable[[i-1]];
Do[
xx=refx +0;
yy=refy + s*Rad*fact;
RingTable[[i]]=iRing;
CxTable[[i]]=xx;
CyTable[[i]]=yy;
If[s < iRing,BordTable[[i,1]]=b++,BordTable[[i,2]]=b++];
i++;
, {s, 1, iRing}];


(*going North-East*)
(*If[ iRing-1\[GreaterEqual] s,*) 
refx=CxTable[[i-1]];
refy=CyTable[[i-1]];
Do[
xx=refx + s*3.*Rad/2.;
yy=refy + s*Rad*fact/2.;
RingTable[[i]]=iRing;
CxTable[[i]]=xx;
CyTable[[i]]=yy;
BordTable[[i,2]]=b++;
i++;
, {s, 1, iRing-1}];
(*];*)

(*nExternal border*)
i= start;

BordTable[[i,1]]=b++;
BordTable[[i,2]]=b++;
i++;

(*Going South -East*)
Do[
If[s< iRing,
BordTable[[i,1]]=b++;
BordTable[[i,2]]=b++;,
BordTable[[i,1]]=b++;
BordTable[[i,2]]=b++;
BordTable[[i,3]]=b++;
];
i++;
,{s, 1, iRing}];

(*Going South *)
Do[
If[s< iRing,
BordTable[[i,2]]=b++;
BordTable[[i,3]]=b++;,
BordTable[[i,2]]=b++;
BordTable[[i,3]]=b++;
BordTable[[i,4]]=b++;
];
i++;
,{s, 1, iRing}];

(*Going South-West *)
Do[
If[s< iRing,
BordTable[[i,3]]=b++;
BordTable[[i,4]]=b++;,
BordTable[[i,3]]=b++;
BordTable[[i,4]]=b++;
BordTable[[i,5]]=b++;
];
i++;
,{s, 1, iRing}];

(*Going North-West *)
Do[
If[s< iRing,
BordTable[[i,4]]=b++;
BordTable[[i,5]]=b++;,
BordTable[[i,4]]=b++;
BordTable[[i,5]]=b++;
BordTable[[i,6]]=b++;
];
i++;
,{s, 1, iRing}];

(*Going North *)
Do[
If[s< iRing,
BordTable[[i,5]]=b++;
BordTable[[i,6]]=b++;,
BordTable[[i,5]]=b++;
BordTable[[i,6]]=b++;
BordTable[[i,1]]=b++;
];
i++;
,{s, 1, iRing}];

(*Going North-East *)
Do[
BordTable[[i,6]]=b++;
BordTable[[i,1]]=b++;
i++;
,{s, 1, iRing-1}];

(*Last border is on the first segment of current ring*)
BordTable[[i-nSegRing,6]]=b++;

, {iRing, 1, nRing}];

(*Neighbours*)
Do[

cx=CxTable[[s]];
cy=CyTable[[s]];
\[Rho]Table=Sqrt[(CxTable-cx)^2+(CyTable -cy)^2];
PATable=Table[If[(CyTable[[s]] -cy==0.) &&(CxTable[[s]]-cx==0.), 0., ArcTan[CxTable[[s]]-cx, CyTable[[s]] -cy]*180./\[Pi]], {s, 1, nSeg}];
aa=Intersection[Flatten[Position[Abs[\[Rho]Table-fact*Rad], _?(# <= 0.1 &)]],  Flatten[Position[Abs[PATable-90.], _?(# <= 0.1 &)]]];
If[Length[aa]> 0,sTable[[s,1]]=aa[[1]]];
bb=Intersection[Flatten[Position[Abs[\[Rho]Table-fact*Rad], _?(# <= 0.1 &)]],  Flatten[Position[Abs[PATable-30.], _?(# <= 0.1 &)]]];
If[Length[bb]> 0,sTable[[s,2]]=bb[[1]]];
cc=Intersection[Flatten[Position[Abs[\[Rho]Table-fact*Rad], _?(# <= 0.1 &)]],  Flatten[Position[Abs[PATable+30.], _?(# <= 0.1 &)]]];
If[Length[cc]> 0,sTable[[s,3]]=cc[[1]]];
dd=Intersection[Flatten[Position[Abs[\[Rho]Table-fact*Rad], _?(# <= 0.1 &)]],  Flatten[Position[Abs[PATable+90.], _?(# <= 0.1 &)]]];
If[Length[dd]> 0,sTable[[s,4]]=dd[[1]]];
ee=Intersection[Flatten[Position[Abs[\[Rho]Table-fact*Rad], _?(# <= 0.1 &)]],  Flatten[Position[Abs[PATable+150.], _?(# <= 0.1 &)]]];
If[Length[ee]> 0,sTable[[s,5]]=ee[[1]]];
ff=Intersection[Flatten[Position[Abs[\[Rho]Table-fact*Rad], _?(# <= 0.1 &)]], Flatten[ Position[Abs[PATable-150.], _?(# <= 0.1 &)]]];
If[Length[ff]> 0,sTable[[s,6]]=ff[[1]]];
, {s, 1, nSeg}];

(*Filling missing borders*)
Do[
If[BordTable[[s, 1]]== -1,BordTable[[s, 1]]=BordTable[[sTable[[s,1]], 4]]];
If[BordTable[[s, 2]]== -1,BordTable[[s, 2]]=BordTable[[sTable[[s,2]], 5]]];
If[BordTable[[s, 3]]== -1,BordTable[[s, 3]]=BordTable[[sTable[[s,3]], 6]]];
If[BordTable[[s, 4]]== -1,BordTable[[s, 4]]=BordTable[[sTable[[s,4]], 1]]];
If[BordTable[[s, 5]]== -1,BordTable[[s, 5]]=BordTable[[sTable[[s,5]], 2]]];
If[BordTable[[s, 6]]== -1,BordTable[[s, 6]]=BordTable[[sTable[[s,6]], 3]]];
, {s, 1, nSeg}];

Table[{iSeg-1, RingTable[[iSeg]],MissingTable[[iSeg]], CxTable[[iSeg]], CyTable[[iSeg]], RingTable[[iSeg]], BordTable[[iSeg]]}, {iSeg, 1, nSeg}]

]




(* ::Subsubsection::Closed:: *)
(*Build hexagonal region*)


(* ::Text:: *)
(*v1*)


BuildHexagonalRegionFunction[xGrid_,yGrid_,xa_,ya_,xb_,yb_]:=Module[{a,b,c,PreRegion,Region},
a=yb-ya;
b=xa-xb;
c=xb*ya -xa*yb;

PreRegion=a*xGrid + b*yGrid + c;
Region=PreRegion;
Do[
Do[
Region[[i,j]]=If[a*xGrid[[i,j]]+ b*yGrid[[i,j]] + c >0.,1,0];
,{i, 1, Dimensions[xGrid][[1]]}];
,{j, 1, Dimensions[xGrid][[2]]}];

Region
]


(* ::Subsubsection::Closed:: *)
(*Build segmented aperture*)


(* ::Text:: *)
(*v1*)


BuildSegmentedApertureFunctionV1b[SegStructure_ , Radius_, GapHexagonFraction_, nPix_]:= Module[{fact,nRing, nSeg,Rad, dim, rx, ry, xx,yy,hx,hy,Pupil,cx, cy, cl, RotMat, p, px, py, nExt, xMin, xMax, yMin,yMax, xxSub, yySub,Segment},

fact=Sqrt[3.];
nRing=Max[SegStructure[[All, 2]]];
nSeg=Length[SegStructure];
Rad=Radius*(1 + GapHexagonFraction);
dim=Ceiling[2*(nRing + 0.5 )*Rad*fact*nPix] +10;
dim=dim + Mod[dim, 2] ;

rx=(dim-1)/2.;
ry=rx;

xx=Table[(i-rx)/nPix, {i, 0, dim-1}, {j, 0, dim-1}];
yy=Table[(j-ry)/nPix, {i, 0, dim-1}, {j, 0, dim-1}];
hx=Radius*{-0.5, 0.5, 1.0, 0.5, -0.5, -1.0, -0.5};
hy=Radius*fact*{0.5, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5};

Pupil=Table[0., {dim}, {dim}];
Do[

If[SegStructure[[s,3]] ==  0,

(*Segment parameters*)
cx=SegStructure[[s,4]];
cy=SegStructure[[s,5]];
cl=0;

(*Apply clocking*)
RotMat={{Cos[cl],-Sin[cl]},{Sin[cl], Cos[cl]}};
p=Table[0.,{Length[hx]}];
Do[
p[[i]]=RotMat.{hx[[i]],hy[[i]]};
,{i, 1, Length[p]}];

(*p=RotMat.{hx,hy};*)
px=p[[All,1]];
py=p[[All,2]];

(*Apply center*)
px=cx + px;
py=cy + py;

(*Sub-window*)
nExt=3;
xMin=Floor[Min[px]*nPix + rx]-nExt;
xMax=Ceiling[Max[px]*nPix+rx]+nExt;
yMin=Floor[Min[py]*nPix + ry]-nExt;
yMax=Ceiling[Max[py]*nPix+ry]+nExt;

xxSub=xx[[xMin;;xMax,yMin;;yMax]];
yySub=yy[[xMin;;xMax,yMin;;yMax]];

Segment=Table[1., {xMax-xMin+1}, {yMax-yMin+1}];
(*0 - North*)
Segment*=BuildHexagonalRegionFunction[xxSub,yySub,px[[1]],py[[1]],px[[2]],py[[2]]];
(*1 - North-East*)
Segment*=BuildHexagonalRegionFunction[xxSub,yySub,px[[2]],py[[2]],px[[3]],py[[3]]];
(*2- South-East*)
Segment*=BuildHexagonalRegionFunction[xxSub,yySub,px[[3]],py[[3]],px[[4]],py[[4]]];
(*3 - South*)
Segment*=BuildHexagonalRegionFunction[xxSub,yySub,px[[4]],py[[4]],px[[5]],py[[5]]];
(*4 - South-West*)
Segment*=BuildHexagonalRegionFunction[xxSub,yySub,px[[5]],py[[5]],px[[6]],py[[6]]];
(*5 - North-West*)
Segment*=BuildHexagonalRegionFunction[xxSub,yySub,px[[6]],py[[6]],px[[1]],py[[1]]];

(*Save Segment*)
Pupil[[xMin;;xMax,yMin;;yMax]]+= Segment;

];

, {s, 1, nSeg}];
Transpose[Pupil]

]


(* ::Subsubsection::Closed:: *)
(*v2 (Test)*)


BuildSegmentedApertureFunctionV2[SegStructure_ , Radius_, GapHexagonFraction_, nPup_]:= Module[{fact,nRing, nSeg,Rad, nZeroPad, dim, nPix, rx, ry, xx,yy,hx,hy,Pupil,cx, cy, cl, RotMat, p, px, py, nExt, xMin, xMax, yMin,yMax, xxSub, yySub,Segment, NewPupil},

fact=Sqrt[3.];
nRing=Max[SegStructure[[All, 2]]];
nSeg=Length[SegStructure];
Rad=Radius*(1 + GapHexagonFraction);
(*dim=Ceiling[2*(nRing + 0.5 )*Rad*fact*nPix] +10;*)
(*dim=dim + Mod[dim, 2] ;*)
nZeroPad=10;
dim=nPup+nZeroPad;
nPix=nPup/(2*(nRing + 0.5 )*Rad*fact);

rx=(dim-1)/2.;
ry=rx;

xx=Table[(i-rx)/nPix, {i, 0, dim-1}, {j, 0, dim-1}];
yy=Table[(j-ry)/nPix, {i, 0, dim-1}, {j, 0, dim-1}];
hx=Radius*{-0.5, 0.5, 1.0, 0.5, -0.5, -1.0, -0.5};
hy=Radius*fact*{0.5, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5};

Pupil=Table[0., {dim}, {dim}];
Do[

If[SegStructure[[s,3]] ==  0,

(*Segment parameters*)
cx=SegStructure[[s,4]];
cy=SegStructure[[s,5]];
cl=0;

(*Apply clocking*)
RotMat={{Cos[cl],-Sin[cl]},{Sin[cl], Cos[cl]}};
p=Table[0.,{Length[hx]}];
Do[
p[[i]]=RotMat.{hx[[i]],hy[[i]]};
,{i, 1, Length[p]}];

(*p=RotMat.{hx,hy};*)
px=p[[All,1]];
py=p[[All,2]];

(*Apply center*)
px=cx + px;
py=cy + py;

(*Sub-window*)
nExt=3;
xMin=Floor[Min[px]*nPix + rx]-nExt;
xMax=Ceiling[Max[px]*nPix+rx]+nExt;
yMin=Floor[Min[py]*nPix + ry]-nExt;
yMax=Ceiling[Max[py]*nPix+ry]+nExt;

xxSub=xx[[xMin;;xMax,yMin;;yMax]];
yySub=yy[[xMin;;xMax,yMin;;yMax]];

Segment=Table[1., {xMax-xMin+1}, {yMax-yMin+1}];
(*0 - North*)
Segment*=BuildHexagonalRegionFunction[xxSub,yySub,px[[1]],py[[1]],px[[2]],py[[2]]];
(*1 - North-East*)
Segment*=BuildHexagonalRegionFunction[xxSub,yySub,px[[2]],py[[2]],px[[3]],py[[3]]];
(*2- South-East*)
Segment*=BuildHexagonalRegionFunction[xxSub,yySub,px[[3]],py[[3]],px[[4]],py[[4]]];
(*3 - South*)
Segment*=BuildHexagonalRegionFunction[xxSub,yySub,px[[4]],py[[4]],px[[5]],py[[5]]];
(*4 - South-West*)
Segment*=BuildHexagonalRegionFunction[xxSub,yySub,px[[5]],py[[5]],px[[6]],py[[6]]];
(*5 - North-West*)
Segment*=BuildHexagonalRegionFunction[xxSub,yySub,px[[6]],py[[6]],px[[1]],py[[1]]];

(*Save Segment*)
Pupil[[xMin;;xMax,yMin;;yMax]]+= Segment;

];

, {s, 1, nSeg}];

(*NewPupil=If[Max[Pupil[[All, 1;;Floor[2.*Radius*GapHexagonFraction]]]] ==0 && Max[Pupil[[All, dim -Floor[2.*Radius*GapHexagonFraction]+1;; dim]]] ==0,
Pupil[[dim/2 -nPup/2+1;; dim/2+nPup/2,dim/2 -nPup/2+1+Floor[2.*Radius*GapHexagonFraction];; dim/2+nPup/2-Floor[2.*Radius*GapHexagonFraction] ]],
Pupil[[dim/2 -nPup/2+1;; dim/2+nPup/2,dim/2 -nPup/2+1;; dim/2+nPup/2 ]]
];
Transpose[NewPupil]*)

NewPupil=Pupil[[dim/2 -nPup/2+1;; dim/2+nPup/2,dim/2 -nPup/2+1;; dim/2+nPup/2]];
Transpose[NewPupil]

]


(* ::Subsection:: *)
(*Spiders generation*)


(* ::Subsubsection::Closed:: *)
(*JWST-like spiders*)


SpidersJWST=Compile[{{nPup, _Real}, {PupilMaskSpidersThickPix, _Real}}, Module[{i,j, Spiders}, 

Spiders=Table[0.,{nPup},{nPup}];
Spiders=Table[ If[(Abs[j-nPup/2-1/2]<= 0.5*PupilMaskSpidersThickPix  && i-nPup/2-1/2>= 0.) || (Abs[Sqrt[3.]/2. (j-(nPup+1)/2.)+1./2. (i-(nPup+1)/2.)]<= 0.5*PupilMaskSpidersThickPix  && i-(nPup+1)/2<=  0.) || (Abs[-(Sqrt[3.]/2.)(j-(nPup+1)/2.)+1./2. (i-(nPup+1)/2.)]<= 0.5*PupilMaskSpidersThickPix  && i-(nPup+1)/2<=  0.), 0., 1.], {i, 1,nPup}, {j, 1, nPup}];
Spiders
]
];



(* ::Subsubsection::Closed:: *)
(*Symetric JWST-like spiders*)


SpidersSymJWST=Compile[{{nPup, _Real}, {PupilMaskSpidersThickPix, _Real}}, Module[{i,j, Spiders}, 

Spiders=Table[0.,{nPup},{nPup}];
Spiders=Table[ If[(Abs[j-nPup/2-1/2]<= 0.5*PupilMaskSpidersThickPix ) || (Abs[Sqrt[3.]/2. (j-(nPup+1)/2.)+1./2. (i-(nPup+1)/2.)]<= 0.5*PupilMaskSpidersThickPix ) || (Abs[-(Sqrt[3.]/2.)(j-(nPup+1)/2.)+1./2. (i-(nPup+1)/2.)]<= 0.5*PupilMaskSpidersThickPix), 0., 1.], {i, 1,nPup}, {j, 1, nPup}];
Spiders
]
];



(* ::Subsubsection::Closed:: *)
(*X-like spiders*)


SpidersX=Compile[{{nPup, _Real}, {PupilMaskSpidersThickPix, _Real}}, Module[{i,j, Spiders}, 

Spiders=Table[0.,{nPup},{nPup}];
Spiders=Table[ If[ (Abs[Sqrt[3.]/2. (j-(nPup+1)/2.)+1./2. (i-(nPup+1)/2.)]<= 0.5*PupilMaskSpidersThickPix ) || (Abs[-(Sqrt[3.]/2.)(j-(nPup+1)/2.)+1./2. (i-(nPup+1)/2.)]<= 0.5*PupilMaskSpidersThickPix), 0., 1.], {i, 1,nPup}, {j, 1, nPup}];
Spiders
]
];



(* ::Subsubsection::Closed:: *)
(*Cross-like spiders*)


SpidersCross=Compile[{{nPup, _Real}, {PupilMaskSpidersThickPix, _Real}}, Module[{i,j, Spiders}, 
Spiders=Table[0.,{nPup},{nPup}];
Spiders=Table[If[Abs[Sqrt[2.]/2. (j-(nPup+1)/2)+Sqrt[2.]/2. (i-(nPup+1)/2)]<= 0.5*PupilMaskSpidersThickPix  || Abs[-(Sqrt[2.]/2.)(j-(nPup+1)/2)+Sqrt[2.]/2. (i-(nPup+1)/2)]<= 0.5*PupilMaskSpidersThickPix , 0., 1.], {i, 1,nPup}, {j, 1, nPup}];
Spiders
]
];



(* ::Subtitle:: *)
(*Parameters*)


(* ::Section:: *)
(*Miscellaneous*)


(* ::Subsection:: *)
(*Units*)


mm=10.^-3;
\[Mu]m=10.^-6;
nm=10.^-9;


(* ::Subsection:: *)
(*Keywords*)


KeywordExportFitsFiles=1;
KeywordQuarterExtraction=0;
KeywordHalfPlaneExtraction=0;


(* ::Section:: *)
(*Pupil*)


(* ::Subsection:: *)
(*Segmented aperture*)


nPup=500;
nRing=3;
Radius=0.5;


(* ::Text:: *)
(*Input for test purposes - parameters for the generation of the segmented geometry (gap size in fraction of the segment size and table of indices of the missing segments)*)


GapHexagonFraction=0.05; 
MissingTable={0};
(*nPix=150;*)
(*AboveKeyword=0;
FineKeyword=1;*)


(* ::Subsection:: *)
(*Spiders*)


(* ::Text:: *)
(*spider shape and symetry*)


SymSpidersKeyword=1;


(* ::Text:: *)
(*spider thickness (wrt Iris AO pupil)*)


(*PupilMaskSpidersThickRatio=0.01;
thick=PupilMaskSpidersThickRatio*nPup;*)


(* ::Section:: *)
(*Magnification factor and error*)


(* ::Text:: *)
(*Magnification between the first plane and the considered plane *)


\[CapitalDelta]\[Gamma]=0.004


\[Gamma]21=0.423;
\[CapitalDelta]\[Gamma]21=\[Gamma]21*\[CapitalDelta]\[Gamma];
\[Gamma]31=1.008;
\[CapitalDelta]\[Gamma]31=\[Gamma]31*\[CapitalDelta]\[Gamma];
\[Gamma]41=0.444;
\[CapitalDelta]\[Gamma]41=\[Gamma]41*\[CapitalDelta]\[Gamma];
\[Gamma]51=0.979;
\[CapitalDelta]\[Gamma]51=\[Gamma]51*\[CapitalDelta]\[Gamma];


(* ::Section:: *)
(*Pupil planes*)


(* ::Text:: *)
(*Number of planes*)


nPlanes=5


(* ::Subsection:: *)
(*P2 - Iris AO*)


(* ::Subsubsection:: *)
(*Physical dimensions*)


(* ::Text:: *)
(*Physical dimensions of the Iris AO*)


P2IrisAOSegmentSize=1.4mm;
P2IrisAOSegmentSideLength=P2IrisAOSegmentSize/2.;

P2IrisAOSegmentGapSize=12\[Mu]m;
P2IrisAOInscribedCircSize=10.*P2IrisAOSegmentSideLength;
P2IrisAOFlatToFlatSize=14 Sqrt[3.]/2. P2IrisAOSegmentSideLength;
P2IrisAOCircumscribedCircSize=Sqrt[P2IrisAOFlatToFlatSize^2+P2IrisAOSegmentSideLength^2];



(* ::Subsubsection:: *)
(*Geometry*)


(* ::Text:: *)
(*Parameters for the generation of the segmented geometry (gap size in fraction of the segment size and table of indices of the missing segments)*)


IrisAOGapHexagonFraction = P2IrisAOSegmentGapSize/P2IrisAOSegmentSideLength


(* ::Input:: *)
(*0.012/0.7*)


(* ::Text:: *)
(*Geoemtry for non missing segment*)


IrisAOMissingTable = {};


(* ::Text:: *)
(*Segment size in pixels*)


nSegPix=P2IrisAOSegmentSize*nPup/P2IrisAOFlatToFlatSize


(* ::Subsubsection:: *)
(*Output sizes*)


"Iris AO inscribed circle size: " <> ToString[PaddedForm[P2IrisAOInscribedCircSize/mm, {5,3}]] <> "mm"
"Iris AO flat-to-flat size: " <> ToString[PaddedForm[P2IrisAOFlatToFlatSize/mm, {5,3}]] <> "mm"
"Iris AO circumscribed circle size: " <> ToString[PaddedForm[P2IrisAOCircumscribedCircSize/mm, {5,3}] ]<> "mm"


(* ::Subsection:: *)
(*P1 - Pupil Mask*)


(* ::Subsubsection:: *)
(*Dimensions*)


(* ::Text:: *)
(*Contour*)


P1PupilMaskSize=19.850mm;
P2PupilMaskSize=P1PupilMaskSize*\[Gamma]21


(* ::Text:: *)
(*Central segment*)


P1PupilMaskCentralSegmentSize=3.600mm;
P2PupilMaskCentralSegmentSize=P1PupilMaskCentralSegmentSize*\[Gamma]21;

PupilMaskCentralSegmentOversizeFactor=P2PupilMaskCentralSegmentSize/P2IrisAOSegmentSize



(* ::Text:: *)
(*Spiders (spider thickness (wrt Iris AO pupil))*)


P1PupilMaskSpidersThick=0.200mm;
PupilMaskSpidersThickPix=P1PupilMaskSpidersThick*nPup/(P2IrisAOFlatToFlatSize/\[Gamma]21)



(* ::Subsubsection:: *)
(*Geometry*)


(* ::Text:: *)
(*Geometry with all the segments*)


PupilMaskContourMissingTable={};


(* ::Text:: *)
(*Geometry with only central segment*)


PupilMaskAllButCentralMissingTable=Range[3*nRing*(nRing+1)];


(* ::Subsubsection:: *)
(*Pupil size fraction*)


(*PupilMaskUndersizeContour=P2PupilMaskSize/P2IrisAOFlatToFlatSize*)
PupilMaskUndersizeContour=P2PupilMaskSize/P2IrisAOCircumscribedCircSize


(* ::Subsubsection:: *)
(*Output sizes*)


"Pupil mask side-to-side contour length in P1: " <>ToString[PaddedForm[P1PupilMaskSize/\[Mu]m, {5,0}]] <> "\[Mu]m"
"Pupil mask central segment side length: "<>ToString[PaddedForm[P1PupilMaskCentralSegmentSize/\[Mu]m, {5,0}]] <> "\[Mu]m"
"Pupil mask strut width: " <> ToString[PaddedForm[P1PupilMaskSpidersThick/\[Mu]m, {5,0}] ]<> "\[Mu]m"


(* ::Subsection:: *)
(*P3 - Apodizer*)


(* ::Subsubsection:: *)
(*Dimensions*)


(* ::Text:: *)
(*Contour*)


P3ApodizerSize=19.725 mm;
P2ApodizerSize=P3ApodizerSize*\[Gamma]21/\[Gamma]31


(* ::Text:: *)
(*Gap*)


P3ApodizerMaskGapSize=0.090mm;
P3IrisAOSegmentGapSize=P2IrisAOSegmentGapSize*\[Gamma]31/\[Gamma]21;
ApodizerMaskGapOversizeFactorWrtIrisAO=P3ApodizerMaskGapSize/P3IrisAOSegmentGapSize;



(* ::Text:: *)
(*Central segment*)


P3ApodizerMaskCentralSegmentSize=3.950mm;
P3PupilMaskCentralSegmentSize=P1PupilMaskCentralSegmentSize*\[Gamma]31;

ApodizerMaskCentralSegmentOversizeFactorWrtPupilMask=P3ApodizerMaskCentralSegmentSize/P3PupilMaskCentralSegmentSize;

(*P3ApodizerMaskCentralSegmentSize=ApodizerMaskCentralSegmentOversizeFactorWrtPupilMask*P3PupilMaskCentralSegmentSize*)

P3IrisAOSegmentSize=P2IrisAOSegmentSize*\[Gamma]31/\[Gamma]21;



(* ::Text:: *)
(*Spiders*)


P3ApodizerMaskSpidersThick=0.350mm;
ApodizerMaskSpidersThickPix=P3ApodizerMaskSpidersThick*nPup/(P2IrisAOFlatToFlatSize*\[Gamma]31/\[Gamma]21)



(* ::Subsection:: *)
(*Fractions used in the AMPL SCRIPT*)


ApodizerMaskUndersizeContourWrtPupilMask=P2ApodizerSize/P2PupilMaskSize;

ApodizerMaskCentralSegmentOversizeFactorWrtIrisAO=P3ApodizerMaskCentralSegmentSize/P3IrisAOSegmentSize;




(* ::Text:: *)
(*Apodizer elements size wrt to IrisAO circumscribed circle size (VALUES USED IN AMPL SCRIPT)*)


ApodizerMaskContourUndersizeFactorWrtIrisAOMask=P3ApodizerSize/(P2IrisAOCircumscribedCircSize*\[Gamma]31/\[Gamma]21)


(* ::Text:: *)
(*Note here that because generating the aperture starts with the regular hexagons (i.e. the IRIS-AO aperture) the central obstruction of the apodizer has to be referenced to the IRIS-AO diameter, which fits in the full array of 500 pixels (actually a couple pixel less initially but it gets truncated at the end)*)


ApodizerMaskCentralSegmentRatioWrtIrisAOMask=P3ApodizerMaskCentralSegmentSize/(P2IrisAOCircumscribedCircSize*\[Gamma]31/\[Gamma]21)


(* ::Text:: *)
(*Same thing here the apodizer spider thickness is referred to the IRIS-AO diameter*)


ApodizerMaskSpidersThickRatioWrtIrisAOMask=P3ApodizerMaskSpidersThick/(P2IrisAOCircumscribedCircSize*\[Gamma]31/\[Gamma]21)


(* ::Text:: *)
(*same thing here for the gap size*)


ApodizerMaskGapRatioWrtIrisAOMask=P3ApodizerMaskGapSize/(P2IrisAOCircumscribedCircSize*\[Gamma]31/\[Gamma]21)


(* ::Subsubsection:: *)
(*Geometry*)


(* ::Text:: *)
(*gap size in fraction of the segment size*)


ApodizerMaskGapHexagonFraction=ApodizerMaskGapOversizeFactorWrtIrisAO*IrisAOGapHexagonFraction


(* ::Text:: *)
(*Geometry with all the segments*)


ApodizerMaskContourMissingTable = {};


(* ::Text:: *)
(*Geometry with only central segment present*)


ApodizerMaskAllButCentralMissingTable = Range[3*nRing*(nRing + 1)];


(* ::Subsubsection:: *)
(*Output sizes*)


"Apodizer mask side-to-side contour length: " <>ToString[PaddedForm[P3ApodizerSize/\[Mu]m, {5,0}]] <> "\[Mu]m"
"Apodizer mask central segment side length: "<>ToString[PaddedForm[P3ApodizerMaskCentralSegmentSize/\[Mu]m, {5,0}] ]<> "\[Mu]m"
"Apodizer mask strut width: " <> ToString[PaddedForm[P3ApodizerMaskSpidersThick/\[Mu]m, {5,0}] ]<> "\[Mu]m"
"Apodizer mask segment gap width: " <> ToString[PaddedForm[P3ApodizerMaskGapSize/\[Mu]m, {5,0}]] <> "\[Mu]m"


(* ::Subsection:: *)
(*P4 - Boston DM 1*)


(* ::Subsubsection:: *)
(*Dimensions*)


P4BostonDM1Size=9.9mm;
nAct=34;
PitchAct=P4BostonDM1Size/nAct;


(* ::Subsubsection:: *)
(*Contour*)


P4ApodizerSize=P3ApodizerSize*\[Gamma]41/\[Gamma]31;
P4IrisAOCircumscribedCircSize=P2IrisAOCircumscribedCircSize*\[Gamma]41/\[Gamma]21;
P4IrisAOFlatToFlatSize=P2IrisAOFlatToFlatSize*\[Gamma]41/\[Gamma]21;


(* ::Subsubsection:: *)
(*Fraction*)


BostonDMUndersizeContourWrtApodizerMask=P4ApodizerSize/P4BostonDM1Size
BostonDM1ContourOversizeFactor=P4BostonDM1Size/P4IrisAOCircumscribedCircSize


(* ::Text:: *)
(*Number of used actuators*)


nPupDM=Ceiling[BostonDM1ContourOversizeFactor*nPup/2]*2


nActUsed=BostonDMUndersizeContourWrtApodizerMask*nAct


(* ::Subsubsection:: *)
(*Output*)


"Actuator pitch: " <>ToString[PaddedForm[PitchAct/\[Mu]m, {5,0}]] <> "\[Mu]m"
"Number of used actuators: " <>ToString[PaddedForm[nActUsed, {4,2}]]



(* ::Subsection:: *)
(*P5 - Lyot stop*)


(* ::Text:: *)
(*Keyword on gray Lyot stop*)


KeywordLyotStopGray=1;


(* ::Subsubsection:: *)
(*Dimensions*)


(* ::Text:: *)
(*Contour*)


P5LyotStopSize=15.9mm;
P5IrisAOInscribedCircSize=P2IrisAOInscribedCircSize*\[Gamma]51/\[Gamma]21;
LyotStopMaskUndersizeContourWrtInscribedCirc=P5LyotStopSize/P5IrisAOInscribedCircSize
(*LyotStopMaskUndersizeContourWrtInscribedCirc = 0.90*)

(0.83*0.90)/0.98


P5IrisAOFlatToFlatSize=P2IrisAOFlatToFlatSize*\[Gamma]51/\[Gamma]21
P5IrisAOCircumscribedCircSize=P2IrisAOCircumscribedCircSize*\[Gamma]51/\[Gamma]21


(10./(14.Sqrt[3.]/2.))*LyotStopMaskUndersizeContourWrtInscribedCirc


(* ::Text:: *)
(*Central segment*)


P5LyotStopMaskCentralSegmentSize=6.800mm;

P5ApodizerMaskCentralSegmentSize=P3ApodizerMaskCentralSegmentSize*\[Gamma]51/\[Gamma]31
P5IriaAOSegmentSize=P2IrisAOSegmentSize*\[Gamma]51/\[Gamma]21
LyotStopMaskCentralSegmentOversizeFactorWrtApodizerMask=P5LyotStopMaskCentralSegmentSize/P5ApodizerMaskCentralSegmentSize



(* ::Text:: *)
(*Spiders*)


P5LyotStopMaskSpidersThick=0.700mm;

LyotStopMaskSpidersThickRatio=P5LyotStopMaskSpidersThick/P5IrisAOCircumscribedCircSize
LyotStopMaskSpidersThickPix=P5LyotStopMaskSpidersThick*nPup/P5IrisAOFlatToFlatSize



(* ::Subsubsection:: *)
(*Fractions*)


LyotStopSizeRatioWrtIrisAOSegment=P5LyotStopSize/P5IriaAOSegmentSize
LyotStopMaskCentralSegmentOversizeFactorWrtIrisAO=P5LyotStopMaskCentralSegmentSize/P5IriaAOSegmentSize


LyotStopMaskContourUndersizeFactorWrtAperture = P2IrisAOInscribedCircSize/P2IrisAOCircumscribedCircSize
LyotStopMaskContourUndersizeFactorWrtIrisAOMask=P2IrisAOInscribedCircSize/P2IrisAOCircumscribedCircSize*LyotStopMaskUndersizeContourWrtInscribedCirc
LyotStopMaskCentralSegmentSizeRatioWrtIrisAOMask=P5LyotStopMaskCentralSegmentSize/P5IrisAOCircumscribedCircSize
LyotStopMaskSpidersThickRatioWrtIrisAOMask=P5LyotStopMaskSpidersThick/P5IrisAOCircumscribedCircSize



(* ::Subsubsection:: *)
(*Output*)


"Lyot stop mask outer edge diameter: " <>ToString[PaddedForm[P5LyotStopSize/\[Mu]m, {5,0}]] <> "\[Mu]m"
"Lyot stop mask inner edge diameter: "<>ToString[PaddedForm[P5LyotStopMaskCentralSegmentSize/\[Mu]m, {5,0}]] <> "\[Mu]m"
"Lyot stop mask strut width: " <> ToString[PaddedForm[P5LyotStopMaskSpidersThick/\[Mu]m, {5,0}]] <> "\[Mu]m"


(* ::Section:: *)
(*FPM*)


(* ::Text:: *)
(*Sampling*)


nFPM=100;


(* ::Text:: *)
(*Mask physical sizes*)


dFPM1=334 \[Mu]m;
dFPM2=455\[Mu]m;


(* ::Text:: *)
(*O8 Focal length*)


fO8=723.3mm;


(* ::Section:: *)
(*Dark zone*)


(* ::Text:: *)
(*boundaries (core, primary dark hole, and secondary dark hole regions)*)


\[Rho]0=3.75;
\[Rho]1=15.0;
\[Rho]2=N@Round[nActUsed Sqrt[2]+\[Rho]1];(*this is calculated for the diagonal assuming 30 actuaotrs in the pupil, including the other side of the DZ (\[Rho]1)*)
\[Rho]2=2*\[Rho]1;(*assumimg we take care of quilting only wiht DM*) 


(* ::Text:: *)
(*contrast in the different regions*)


cCoreRegion=10.^-6;(*100-1000 dynamic range if we have 1e8 to 1e7 final contrast*)
cDarkHoleRegion=10.^-8;
cSecondaryRegion=10.^-6;(*since not the right way to do it, we don't push too hard here for now.*)


(* ::Section:: *)
(*Wavelength*)


(* ::Subsection:: *)
(*Laser source*)


\[Lambda]MCLS1=639.3nm
\[Lambda]MCLS2=640.3nm
\[Lambda]0=(\[Lambda]MCLS1+\[Lambda]MCLS2)/2.


(* ::Subsection:: *)
(*Spectral band and sampling*)


bandwidth=0.10;
n\[Lambda]=3;


(* ::Section:: *)
(*Tests*)


(* ::Subsection:: *)
(*Telescope aperture (tests)*)


SegStructureTest=SegStructureFunction[nRing, Radius, GapHexagonFraction, MissingTable];


SegmentedPupilTest=BuildSegmentedApertureFunctionV2[SegStructureTest , Radius, GapHexagonFraction, nPup];
(*ArrayPlot[SegmentedPupil, ColorFunction\[Rule] GrayLevel]*)


Spiders=If[PupilMaskSpidersThickPix ==0.,
Table[1., {nPup}, {nPup}],
If[SymSpidersKeyword==1, SpidersX[nPup, PupilMaskSpidersThickPix], SpidersJWST[nPup, PupilMaskSpidersThickPix]]
];
(*ArrayPlot[Spiders, ColorFunction\[Rule] GrayLevel]*)


SegmentedApertureTest=SegmentedPupilTest*Spiders;
ArrayPlot[SegmentedApertureTest,ColorFunction-> GrayLevel]


(* ::Section:: *)
(*Pupil generation*)


(* ::Subsection:: *)
(*P2 - Iris-AO *)


(* ::Subsubsection:: *)
(*Structure*)


IrisAOGapHexagonFractionSegStructure = SegStructureFunction[nRing, Radius, IrisAOGapHexagonFraction, IrisAOMissingTable];


(* ::Subsubsection:: *)
(*Geometry*)


IrisAOPupilMask=BuildSegmentedApertureFunctionV2[IrisAOGapHexagonFractionSegStructure , Radius, IrisAOGapHexagonFraction, nPup];


(* ::Subsubsection:: *)
(*Display*)


ArrayPlot[IrisAOPupilMask, ColorFunction -> GrayLevel]


(* ::Subsection:: *)
(*P1 - Pupil mask*)


(* ::Subsubsection:: *)
(*Structure*)


PupilMaskContourSegStructure=SegStructureFunction[nRing, Radius, 0., PupilMaskContourMissingTable];
PupilMaskAllButCentralSegStructure=SegStructureFunction[nRing, Radius, 0., PupilMaskAllButCentralMissingTable];


(* ::Subsubsection:: *)
(*Geometry*)


(* ::Text:: *)
(*Pupil mask contour (undersize wrt IrisAO)*)


Pre2PupilMaskContour=BuildSegmentedApertureFunctionV2[PupilMaskContourSegStructure , Radius, 0., nPup];


PrePupilMaskContour=Abs[Resize[IFFT[Resize[FFT[Resize[Pre2PupilMaskContour, 2*nPup]],Floor[PupilMaskUndersizeContour*nPup]*2]], nPup]];
PupilMaskContour=Round[PrePupilMaskContour/Max[PrePupilMaskContour]];


PupilMaskUndersizeContourFinal=Floor[PupilMaskUndersizeContour*nPup]*2/(2.*nPup)


(* ::Text:: *)
(*Central obstruction (oversize wrt IrisAO)*)


Pre2PupilMaskAllButCentral=BuildSegmentedApertureFunctionV2[PupilMaskAllButCentralSegStructure , Radius, 0., nPup];


nBis=nPup/2+ Mod[nPup/2,2]
nTer=Ceiling[PupilMaskCentralSegmentOversizeFactor* nBis]+ Mod[Ceiling[PupilMaskCentralSegmentOversizeFactor* nBis], 2]

PrePupilMaskAllButCentral=Abs[Resize[IFFT[Resize[FFT[Resize[Pre2PupilMaskAllButCentral, nBis]],nTer]], nPup]];
PupilMaskAllButCentral=Round[PrePupilMaskAllButCentral/Max[PrePupilMaskAllButCentral]];



(* ::Text:: *)
(*Spiders*)


PupilMaskSpiders=If[PupilMaskSpidersThickPix ==0.,
Table[1., {nPup}, {nPup}],
If[SymSpidersKeyword==1, SpidersX[nPup, PupilMaskSpidersThickPix], SpidersJWST[nPup, PupilMaskSpidersThickPix]]
];


(* ::Subsubsection:: *)
(*Pupil mask - display*)


PupilMask=PupilMaskSpiders*PupilMaskContour*(1-PupilMaskAllButCentral);


ArrayPlot[PupilMask, ColorFunction-> GrayLevel]


(* ::Subsubsection:: *)
(*Difference between IrisAO & Pupil Mask *)


ArrayPlot[IrisAOPupilMask-PupilMask , ColorFunction-> GrayLevel]


(* ::Subsection:: *)
(*P3 - Apodizer*)


(* ::Subsubsection:: *)
(*Structure*)


ApodizerMaskSegmentationStructure=SegStructureFunction[nRing, Radius, ApodizerMaskGapHexagonFraction, ApodizerMaskContourMissingTable];
ApodizerMaskContourSegStructure=SegStructureFunction[nRing, Radius, 0., ApodizerMaskContourMissingTable];
ApodizerMaskAllButCentralSegStructure=SegStructureFunction[nRing, Radius, 0., ApodizerMaskAllButCentralMissingTable];


(* ::Subsubsection:: *)
(*Geometry*)


(* ::Text:: *)
(*Segmentation*)


ApodizerMaskSegmentationMask=BuildSegmentedApertureFunctionV2[ApodizerMaskSegmentationStructure , Radius, ApodizerMaskGapHexagonFraction, nPup];


(* ::Text:: *)
(*Apodizer mask contour (undersize wrt Iris AO and Pupil Mask)*)


Pre2ApodizerMaskContour=BuildSegmentedApertureFunctionV2[ApodizerMaskContourSegStructure , Radius, 0., nPup];

PreApodizerMaskContour=Abs[Resize[IFFT[Resize[FFT[Resize[Pre2ApodizerMaskContour, 2*nPup]],Floor[ApodizerMaskContourUndersizeFactorWrtIrisAOMask*nPup]*2]], nPup]];
ApodizerMaskContour=Round[PreApodizerMaskContour/Max[PreApodizerMaskContour]];


ArrayPlot[ApodizerMaskContour-IrisAOPupilMask, ColorFunction -> GrayLevel]


ApodizerMaskUndersizeContourFinal=Floor[ApodizerMaskContourUndersizeFactorWrtIrisAOMask*2*nPup]/(2.*nPup)


(* ::Text:: *)
(*Central segment (oversize wrt Iris AO and Pupil Mask)*)


Pre2ApodizerMaskAllButCentral=BuildSegmentedApertureFunctionV2[ApodizerMaskAllButCentralSegStructure, Radius, 0., nPup];

nBis2=nPup/2+ Mod[nPup/2,2]
nTer2=Ceiling[ApodizerMaskCentralSegmentOversizeFactorWrtIrisAO* nBis2]+ Mod[Ceiling[ApodizerMaskCentralSegmentOversizeFactorWrtIrisAO* nBis2], 2]

PreApodizerMaskAllButCentral=Abs[Resize[IFFT[Resize[FFT[Resize[Pre2ApodizerMaskAllButCentral, nBis2]],nTer2]], nPup]];
ApodizerMaskAllButCentral=Round[PreApodizerMaskAllButCentral/Max[PreApodizerMaskAllButCentral]];


(* ::Text:: *)
(*Spiders*)


ApodizerMaskSpiders=If[ApodizerMaskSpidersThickPix ==0.,
Table[1., {nPup}, {nPup}],
If[SymSpidersKeyword==1, SpidersX[nPup, ApodizerMaskSpidersThickPix], SpidersJWST[nPup, ApodizerMaskSpidersThickPix]]
];


(* ::Subsubsection:: *)
(*Apodizer mask - display*)


ApodizerMask=ApodizerMaskSpiders*ApodizerMaskSegmentationMask*ApodizerMaskContour*(1-ApodizerMaskAllButCentral);


ArrayPlot[ApodizerMask, ColorFunction -> GrayLevel]


(* ::Subsubsection:: *)
(*Difference between Pupil Mask & Apodizer Mask*)


ArrayPlot[IrisAOPupilMask- ApodizerMask, ColorFunction-> GrayLevel]


(* ::Subsection:: *)
(*P4 - Boston DM*)


(* ::Subsubsection:: *)
(*Geometry*)


(* ::Text:: *)
(*Contour*)


ResizeBostonDMMaskContour=Cr\[EAcute]eDiaphragmeCenter[BostonDM1ContourOversizeFactor*nPup/2, nPupDM];


(* ::Subsubsection:: *)
(*Boston DM - display*)


(*BostonDMMask=BostonDMMaskContour;*)
ResizeBostonDMMask=ResizeBostonDMMaskContour;
ArrayPlot[ResizeBostonDMMask, ColorFunction -> GrayLevel]


(* ::Subsubsection:: *)
(*Difference between Apodizer & Boston DM - display*)


ArrayPlot[Resize[ApodizerMask, nPupDM]- ResizeBostonDMMask, ColorFunction -> GrayLevel]


(* ::Subsubsection:: *)
(*Difference between Iris AO & Apodizer - display*)


ArrayPlot[Resize[IrisAOPupilMask, nPupDM]- ResizeBostonDMMask, ColorFunction -> GrayLevel, Frame -> None]


(* ::Subsection:: *)
(*P5 - Lyot stop*)


(* ::Subsubsection:: *)
(*Geometry*)


(* ::Text:: *)
(*Contour*)


LyotStopMaskContourBW=Cr\[EAcute]eDiaphragmeCenter[LyotStopMaskContourUndersizeFactorWrtIrisAOMask*nPup/2, nPup]-Cr\[EAcute]eDiaphragmeCenter[LyotStopMaskCentralSegmentOversizeFactorWrtIrisAO*nSegPix/2., nPup];


LyotStopMaskContourGray=Cr\[EAcute]eDiaphragmeLisseCenter[LyotStopMaskContourUndersizeFactorWrtIrisAOMask*nPup/2, nPup, 11]-Cr\[EAcute]eDiaphragmeLisseCenter[LyotStopMaskCentralSegmentOversizeFactorWrtIrisAO*nSegPix/2., nPup, 11];


ArrayPlot[LyotStopMaskContourGray]


nLyotPup0=LyotStopMaskContourUndersizeFactorWrtIrisAOMask*nPup


(* ::Text:: *)
(*Spiders*)


LyotStopMaskSpiders=If[LyotStopMaskSpidersThickPix ==0.,
Table[1., {nPup}, {nPup}],
If[SymSpidersKeyword==1, SpidersX[nPup, LyotStopMaskSpidersThickPix], SpidersJWST[nPup,LyotStopMaskSpidersThickPix]]
];


(* ::Subsubsection:: *)
(*Lyot Stop mask - display*)


LyotStopMaskBW=LyotStopMaskSpiders*LyotStopMaskContourBW;
LyotStopMaskGray=LyotStopMaskSpiders*LyotStopMaskContourGray;


ArrayPlot[LyotStopMaskBW, ColorFunction -> GrayLevel]
ArrayPlot[LyotStopMaskGray-LyotStopMaskBW, ColorFunction -> GrayLevel]


(* ::Subsubsection:: *)
(*Difference between Lyot stop & apodizer Mask *)


ArrayPlot[LyotStopMaskBW-ApodizerMask, ColorFunction -> GrayLevel]
ArrayPlot[LyotStopMaskGray-ApodizerMask, ColorFunction -> GrayLevel]


(* ::Section:: *)
(*Zeropadding removal*)


(* ::Subsubsection:: *)
(*Removal of zero padding  *)


iApodFind=1;While[Max[ApodizerMask[[iApodFind, All]]]==  0.,iApodFind++];
iApodFindBis=nPup;While[Max[ApodizerMask[[iApodFindBis, All]]]==  0.,iApodFindBis--];
jApodFind=1;While[Max[ApodizerMask[[All, jApodFind]]]==  0.,jApodFind++];
jApodFindBis=nPup;While[Max[ApodizerMask[[All, jApodFindBis]]]==  0.,jApodFindBis--];



ijApodFind=Min[{iApodFind, jApodFind}];
ijApodFindBis=Max[{iApodFindBis, jApodFindBis}];
ApodizerMaskBis=ApodizerMask[[ijApodFind;;ijApodFindBis, ijApodFind;;ijApodFindBis]];
LyotStopMaskBisBW=LyotStopMaskBW[[ijApodFind;;ijApodFindBis, ijApodFind;;ijApodFindBis]];
LyotStopMaskBisGray=LyotStopMaskGray[[ijApodFind;;ijApodFindBis, ijApodFind;;ijApodFindBis]];



nApodPup=Length[ApodizerMaskBis]


ArrayPlot[ApodizerMaskBis, ColorFunction -> GrayLevel]
ArrayPlot[LyotStopMaskBisBW,ColorFunction -> GrayLevel]
ArrayPlot[LyotStopMaskBisGray,ColorFunction -> GrayLevel]


(* ::Input:: *)
(**)


iLyotFind=1;While[Max[LyotStopMaskBW[[iLyotFind, All]]]==  0.,iLyotFind++];
iLyotFindBis=nPup;While[Max[LyotStopMaskBW[[iLyotFindBis, All]]]==  0.,iLyotFindBis--];
jLyotFind=1;While[Max[LyotStopMaskBW[[All, jLyotFind]]]==  0.,jLyotFind++];
jLyotFindBis=nPup;While[Max[LyotStopMaskBW[[All, jLyotFindBis]]]==  0.,jLyotFindBis--];


nLyotPup=Max[{iLyotFindBis-iLyotFind+1,jLyotFindBis-jLyotFind+1}]


(* ::Input:: *)
(**)


(* ::Section:: *)
(*FPM relative dimensions*)


(*Dbeam=P4ApodizerSize*)
Dbeam=P4ApodizerSize*nApodPup/(ApodizerMaskContourUndersizeFactorWrtIrisAOMask*nPup)(*P4BostonDM1Size*nApodPup/nPupDM*)
P4ApodizerSize
Fratio=fO8/Dbeam


dMask1=dFPM1/(\[Lambda]0*Fratio)
dMask2=dFPM2/(\[Lambda]0*Fratio)


MMask1=dMask1/2.;
MMask2=dMask2/2.;


(* ::Text:: *)
(*Undersizing of the Lyot stop with respect to the apodizer (not the Iris AO)*)


UndersizePup=N[nLyotPup0/nApodPup]


(* ::Subtitle:: *)
(*Export files*)


(* ::Section:: *)
(*File nomenclature*)


(* ::Subsection:: *)
(*Roots*)


(* ::Subsubsection:: *)
(*File type*)


RootTestbed="HiCAT";
RootNameList={"-Aper",  "-Lyot"};
RootPupilGeometry={"_F", "_H", "_Q"};
RootQuarterSize="-N" <> PaddingForFileNames[Round[nApodPup/2], 4];
RootHalfSize="-N" <> PaddingForFileNames[Round[nApodPup/2], 4];
RootFullSize="-N" <> PaddingForFileNames[nApodPup, 4];


RootFileTypeAperQ=RootTestbed <> RootNameList[[1]] <> RootPupilGeometry[[3]] <> RootQuarterSize;
RootFileTypeLyotQ=RootTestbed <> RootNameList[[2]] <> RootPupilGeometry[[3]] <> RootQuarterSize;

RootFileTypeAperH=RootTestbed <> RootNameList[[1]] <> RootPupilGeometry[[2]] <> RootHalfSize;
RootFileTypeLyotH=RootTestbed <> RootNameList[[2]] <> RootPupilGeometry[[2]] <> RootHalfSize;

RootFileTypeAperF=RootTestbed <> RootNameList[[1]] <> RootPupilGeometry[[1]] <> RootFullSize;
RootFileTypeLyotF=RootTestbed <> RootNameList[[2]] <> RootPupilGeometry[[1]] <> RootFullSize;


(* ::Subsubsection::Closed:: *)
(*Aperture type*)


RootAperType="_Hex3";
RootAperContour="-Ctr" <> PaddingForFileNames[Round[ApodizerMaskContourUndersizeFactorWrtIrisAOMask*1000.], 4];
RootAperCentralSegment="-Obs" <> PaddingForFileNames[Round[ApodizerMaskCentralSegmentRatioWrtIrisAOMask*1000.], 4];
RootAperSpiders="-Sp" <> If[SymSpidersKeyword==1,"X", "Y"] <> PaddingForFileNames[Round[ApodizerMaskSpidersThickRatioWrtIrisAOMask*1000.], 4];
RootAperGap="-Gap" <> PaddingForFileNames[Round[ApodizerMaskGapRatioWrtIrisAOMask*1000.], 4];



RootApertureType=RootAperType <> RootAperContour <> RootAperCentralSegment <> RootAperSpiders <>RootAperGap;


(* ::Subsubsection::Closed:: *)
(*FPM*)


RootFPMSizeList="_FPM" <> PaddingForFileNames[Round[ dMask2*1000.], 4];
RootFPMQuarterSize="-M" <> PaddingForFileNames[Round[nFPM/2], 3];


RootFPM=RootFPMSizeList <> RootFPMQuarterSize;


(* ::Subsubsection::Closed:: *)
(*Lyot stop*)


RootLyotType="_LS-Ann-" <> If[KeywordLyotStopGray==1, "gy", "bw"]; 
RootLyotStopID="-ID" <> PaddingForFileNames[Round[LyotStopMaskCentralSegmentSizeRatioWrtIrisAOMask*1000.], 4];
RootLyotStopOD="-OD" <> PaddingForFileNames[Round[LyotStopMaskContourUndersizeFactorWrtIrisAOMask*1000.], 4];
RootLyotSpiders="-Sp" <> If[SymSpidersKeyword==1,"X", "Y"] <> PaddingForFileNames[Round[LyotStopMaskSpidersThickRatioWrtIrisAOMask*1000.], 4];


RootLyot=RootLyotType <> RootLyotStopID <> RootLyotStopOD <> RootLyotSpiders;


(* ::Section:: *)
(*FITS files*)


(* ::Subsection::Closed:: *)
(*Quarter pupil extraction and export*)


If[KeywordQuarterExtraction==1,
ApodizerMaskBisQuarter =ApodizerMaskBis[[nApodPup/2+1;; nApodPup,nApodPup/2+1;; nApodPup ]];
LyotStopMaskBisQuarterBW=LyotStopMaskBisBW[[nApodPup/2+1;; nApodPup,nApodPup/2+1;; nApodPup ]];
LyotStopMaskBisQuarterGray=LyotStopMaskBisGray[[nApodPup/2+1;; nApodPup,nApodPup/2+1;; nApodPup ]];
]


(*ArrayPlot[ApodizerMaskBisQuarter, ColorFunction \[Rule] GrayLevel]
ArrayPlot[LyotStopMaskBisQuarterBW, ColorFunction \[Rule] GrayLevel]
ArrayPlot[LyotStopMaskBisQuarterGray, ColorFunction \[Rule] GrayLevel]*)


(* ::Subsection::Closed:: *)
(*Half pupil extraction*)


If[KeywordHalfPlaneExtraction==1,
ApodizerMaskBisHalf =ApodizerMaskBis[[All,nApodPup/2+1;; nApodPup ]];
LyotStopMaskBisHalfBW=LyotStopMaskBisBW[[All,nApodPup/2+1;; nApodPup ]];
LyotStopMaskBisHalfGray=LyotStopMaskBisGray[[All,nApodPup/2+1;; nApodPup ]];
]


(*ArrayPlot[ApodizerMaskBisHalf, ColorFunction \[Rule] GrayLevel]
ArrayPlot[LyotStopMaskBisHalfBW, ColorFunction \[Rule] GrayLevel]
ArrayPlot[LyotStopMaskBisHalfGray, ColorFunction \[Rule] GrayLevel]*)


(* ::Subsection:: *)
(*Filenames*)


(* ::Subsubsection::Closed:: *)
(*Aperture*)


FilenameApertureQ="apertures/"<>RootFileTypeAperQ <> RootApertureType <> ".fits";
FilenameApertureH="apertures/"<>RootFileTypeAperH <> RootApertureType <> ".fits";
FilenameApertureF="apertures/"<>RootFileTypeAperF <> RootApertureType <> ".fits";


(* ::Subsubsection::Closed:: *)
(*Lyot stop*)


FilenameLyotStopQ="lyot_stops/"<>RootFileTypeLyotQ <> RootLyot <> ".fits"
FilenameLyotStopH="lyot_stops/"<>RootFileTypeLyotH <> RootLyot <> ".fits"
FilenameLyotStopF="lyot_stops/"<>RootFileTypeLyotF <> RootLyot <> ".fits"


(* ::Subsection:: *)
(*Filepaths*)


(* ::Text:: *)
(*Quarter extraction*)


PupilMaskForApodizerFilePathQuarter=FileNameJoin[{HiCATDirectory, FilenameApertureQ}];
LyotStopMaskFilePathQuarter=FileNameJoin[{HiCATDirectory, FilenameLyotStopQ}];


(* ::Text:: *)
(*Half extraction*)


PupilMaskForApodizerFilePathHalf=FileNameJoin[{HiCATDirectory, FilenameApertureH}];
LyotStopMaskFilePathHalf=FileNameJoin[{HiCATDirectory, FilenameLyotStopH}];


(* ::Text:: *)
(*Full pupil*)


PupilMaskForApodizerFilePathForFull=FileNameJoin[{HiCATDirectory, FilenameApertureF}]
LyotStopMaskFilePathFull=FileNameJoin[{HiCATDirectory, FilenameLyotStopF}]


(* ::Subsection:: *)
(*Export FITS files*)


(* ::Subsubsection:: *)
(*Quarter extraction*)


If[KeywordQuarterExtraction==1,
Export[PupilMaskForApodizerFilePathQuarter, ApodizerMaskBisQuarter];
Export[LyotStopMaskFilePathQuarter, LyotStopMaskBisQuarterBW];
]


(* ::Subsubsection:: *)
(*Half extraction*)


If[KeywordHalfPlaneExtraction==1,
Export[PupilMaskForApodizerFilePathHalf, ApodizerMaskBisHalf];
Export[LyotStopMaskFilePathHalf, If[KeywordLyotStopGray==1,LyotStopMaskBisHalfGray, LyotStopMaskBisHalfBW]];
]


(* ::Subsubsection:: *)
(*Full pupil*)


Export[PupilMaskForApodizerFilePathForFull, ApodizerMaskBis]
Export[LyotStopMaskFilePathFull,LyotStopMaskBisBW]
























