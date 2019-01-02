#!/usr/bin/env wolframscript
(* ::Package:: *)

(* ::Title:: *)
(*PSF generation for APLC/SP with AMPL files for HiCAT*)


(* ::Section::Initialization:: *)
(*Definitions*)


(* ::Subsection::Initialization:: *)
(*Initializations*)


(* ::Text:: *)
(*Package loading *)


(* ::Input::Initialization:: *)
PackageVersion="5.0.9";
Init[PackageVersion]


(* ::Input::Initialization:: *)
Needs["Developer`"]


(* ::Input::Initialization:: *)
Off[SFT::"NewPlan"]
Off[ToLyotPlane::"NewPlan"]
SetOptions[ListLogPlot,Joined->True,Frame->True,Axes->None,PlotStyle->AbsoluteThickness[1.5]];


(* ::Section:: *)
(*Directory*)


(* ::Subsection:: *)
(*FileDirectoryHiCATFiles definition*)


(* ::Text:: *)
(*Directory of the HiCAT files (aperture, apodizer, Lyot stop)*)


(*FileDirectoryHiCATFiles=ParentDirectory[NotebookDirectory[]]<>"/optimization/";*)


FileDirectoryHiCATFiles="/user/kstlaurent/SCDA_2";


(* ::Section:: *)
(*Functions*)


(* ::Subsection::Closed:: *)
(*Padding For FileName function*)


(* ::Text:: *)
(*Function to pad string numbers and complete with zeros in front *)


PaddingForFileNames[x_,ndigit_]:=PaddedForm[x,ndigit,NumberPadding->{"0",""},NumberSigns->{"",""}]//ToString


(* ::Subsection::Closed:: *)
(*Full pupil generation*)


(* ::Subsubsection::Closed:: *)
(*From quarter pupil*)


FullPupilFromQuarterGenerationFunction[ArrayQuarter_, nPup_]:= Module[{ArrayFromQuarter, ArrayOtherQuarter, ArrayOtherHalfQ, ArrayOtherQuarter2},
ArrayFromQuarter=Table[0., {i, 1, nPup}, {j, 1, nPup}];
ArrayFromQuarter[[nPup/2+1;;nPup, nPup/2+1;;nPup]]=ArrayQuarter;
ArrayOtherQuarter=Table[ArrayQuarter[[i, nPup/2+1 - j]], {i, 1, nPup/2},{j, 1, nPup/2}];
ArrayFromQuarter[[ nPup/2+1;;nPup,1;;nPup/2]]=ArrayOtherQuarter;
ArrayOtherHalfQ=Table[ArrayFromQuarter[[ i, j]], {i, nPup/2+1, nPup},{j, 1, nPup}];
ArrayOtherQuarter2=Table[ ArrayOtherHalfQ[[nPup/2+1-i, j]], {i, 1, nPup/2},{j, 1, nPup}];
ArrayFromQuarter[[1;;nPup/2, All]]=ArrayOtherQuarter2;
ArrayFromQuarter
]


(* ::Subsubsection::Closed:: *)
(*From half pupil*)


FullPupilFromHalfGenerationFunction[PupilHalf_, nPup_]:=Module[{PupilFromHalf,PupilOtherHalf},
PupilFromHalf=Table[0., {i, 1, nPup}, {j, 1, nPup}];
PupilFromHalf[[All, nPup/2+1;;nPup]]=PupilHalf;
PupilOtherHalf=Table[PupilHalf[[i, nPup/2+1 - j]], {i, 1, nPup},{j, 1, nPup/2}];
PupilFromHalf[[ All,1;;nPup/2]]=PupilOtherHalf;
PupilFromHalf
]


(* ::Subsection::Closed:: *)
(*Array Size Increase*)


(* ::Subsubsection::Closed:: *)
(*Size increase*)


(* ::Text:: *)
(*Function to oversize a given array by an integer factor and inclusion of the values from the original array*)


SizeIncrease=Compile[{{image,_Real,2},{coeff,_Integer}},Module[{i,j,i0,j0,n,res},
n=Length[image]*coeff;
res=Table[0.,{n},{n}];
Do[
Do[
i0=Quotient[i+coeff-1,coeff];
j0=Quotient[j+coeff-1,coeff];
res[[i,j]]=image[[i0, j0]];
,{i,1,n}];
,{j,1,n}];
res
]
];


(* ::Subsubsection::Closed:: *)
(*Size increase and round*)


(* ::Text:: *)
(*Function to oversize a given array by an integer factor and inclusion of the rounded values from the original array*)


SizeIncreaseAndRound=Compile[{{image,_Real,2},{coeff,_Integer}},Module[{i,j,i0,j0,n,res},
n=Length[image]*coeff;
res=Table[0.,{n},{n}];
Do[
Do[
i0=Quotient[i+coeff-1,coeff];
j0=Quotient[j+coeff-1,coeff];
res[[i,j]]=Round[image[[i0, j0]]];
,{i,1,n}];
,{j,1,n}];
res
]
];


(* ::Subsubsection::Closed:: *)
(*Size increase and binarization*)


(* ::Text:: *)
(*Function to oversize a given array by an integer factor and inclusion of the values from the original array after binarization*)


SizeIncreaseAndBinarization=Compile[{{image,_Real,2},{coeff,_Integer}},Module[{i,j,i0,j0,n,res, frac, i1, j1, k1,test},
n=Length[image]*coeff;
res=Table[0.,{n},{n}];
frac=1./(coeff^2+1);
Do[
Do[
i0=Quotient[i+coeff-1,coeff];
j0=Quotient[j+coeff-1,coeff];

i1=Mod[i+coeff-1, coeff];
j1=Mod[j+coeff-1, coeff];
k1=i1*coeff+j1+1;
test=Quotient[image[[i0, j0]],frac ];
res[[i,j]]=If[ test >=  k1,1.,0.];
,{i,1,n}];
,{j,1,n}];
res
]
];


(* ::Subsubsection::Closed:: *)
(*Size increase and random dither algorithm*)


(* ::Text:: *)
(*Function to oversize a given array by an integer factor and inclusion of values from the original array after random dither algorithm*)


SizeIncreaseAndRandomDither=Compile[{{image,_Real,2},{coeff,_Integer}},Module[{i,j,i0,j0,n,res},
n=Length[image]*coeff;
res=Table[0.,{n},{n}];
Do[
Do[
i0=Quotient[i+coeff-1,coeff];
j0=Quotient[j+coeff-1,coeff];
res[[i,j]]=If[RandomVariate[UniformDistribution[{0,1}]] <= image[[i0, j0]], 1., 0.];
,{i,1,n}];
,{j,1,n}];
res
]
];


(* ::Subsubsection::Closed:: *)
(*Size increase and error diffusion algorithm*)


(* ::Text:: *)
(*Function to oversize a given array by an integer factor and inclusion of values from the original array after error diffusion algorithm (e.g. Dorrer & Zuegel 2007)*)


SizeIncreaseAndErrorDiffusion=Compile[{{image,_Real,2},{coeff,_Integer}},Module[{i,j,n,res,err, SuperImage},
n=Length[image]*coeff;
res=Table[0.,{n},{n}];
err=Table[0.,{n},{n}];
SuperImage=SizeIncrease[image, coeff];
(*Print[Dimensions[SuperImage]];*)
Do[
Do[
If[SuperImage[[i,j]] > 0.5,res[[i,j]]=1., res[[i, j]]=0.];
err[[i,j]]=res[[i,j]]-SuperImage[[i,j]];
If[ i+1 <= n && j-1 <= n  && j-1 >= 1 , SuperImage[[i+1, j-1]]=SuperImage[[i+1, j-1]]+-3./16. *err[[i,j]]];
If[ i+1 <= n && j+0 <= n , SuperImage[[i+1, j+0]]=SuperImage[[i+1, j+0]]+-5./16. *err[[i,j]]];
If[ i+1 <= n && j+1 <= n , SuperImage[[i+1, j+1]]=SuperImage[[i+1, j+1]]+-1./16. *err[[i,j]]];
If[ i +0<= n && j+1 <= n , SuperImage[[i+0, j+1]]=SuperImage[[i+0, j+1]]+-7./16. *err[[i,j]]];
,{j,1,n}];
,{i,1,n}];
res
]
];


(* ::Subsection::Closed:: *)
(*ToLyotPlane center*)


(* ::Text:: *)
(*ToLyotPlaneCenter function reproduces the ToLyotPlane function that is used in Coronagraphy package to make propagation for pupils centered between four pixels*)


(* ::Subsubsection::Closed:: *)
(*PlaneToLyotPlaneCenter*)


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
(*ToLyotPlaneCenter*)


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



(* ::Subsection::Closed:: *)
(*Functions for Airy throughput computation*)


(* ::Subsubsection::Closed:: *)
(*Make circ*)


(* ::Text:: *)
(*Make centered circle with given radius and array size as parameters*)


MakeCirc=Compile[{i,j,\[Rho],n0},If[Sqrt[(i-n0)^2+(j-n0)^2]<=\[Rho] ,1,0]];
Cr\[EAcute]eCirc[\[Rho]1_, n_]:=Module[{i,j,n0},
									    n0=n/2+1/2;
									    Table[MakeCirc[i,j,\[Rho]1,n0],{i,n},{j,n}]
									 ];


(* ::Subsubsection::Closed:: *)
(*Make decentered circ*)


(* ::Text:: *)
(*Make offset circle with given radius, array size, and offset values wrt to the center as parameters *)


MakeDecenteredCirc=Compile[{i,j,\[Rho],n1, n2},If[Sqrt[(i-n1)^2+(j-n2)^2]<=\[Rho] ,1,0]];
Cr\[EAcute]eDecenteredCirc[\[Rho]1_, n_, xdc_, ydc_]:=Module[{i,j,n1,n2},
									    n1=n/2+1/2+ydc;
									n2=n/2+1/2+xdc;
									    Table[MakeDecenteredCirc[i,j,\[Rho]1,n1, n2],{i,n},{j,n}]
									 ];


(* ::Subsection::Closed:: *)
(*Contrast List Plot*)


ContrastListPlot[PreListPlot_, MMaskProj_, \[Rho]0_, \[Rho]1_, contrast_,ContrastPlotLabel_]:=Module[{},
ListPlot[PreListPlot,Joined->True, PlotRange-> {{-0.5,20.5}, {-1.8, -12.2}}, ImageSize-> Large,LabelStyle->Directive[FontFamily->"Helvetica", FontSize-> 12.], 
PlotLabel->ContrastPlotLabel,
FrameLabel-> {"Angular separation in \!\(\*SubscriptBox[\(\[Lambda]\), \(0\)]\)/\!\(\*SubscriptBox[\(D\), \(Apod\)]\)", "Normalized intensity in log scale"},
FrameStyle-> AbsoluteThickness[1.1],
PlotStyle->If[Length[PreListPlot] ==1, Directive[ColorData["Rainbow", 0],AbsoluteThickness[2.0]], Table[Directive[ColorData["Rainbow",i/(Length[PreListPlot]-1)], AbsoluteThickness[2.0]],{i,0,Length[PreListPlot]-1}]],
Epilog ->{FontSize -> 15.,AbsoluteThickness[2.0],{Dashed,Blue, Line[{{{\[Rho]0, 0}, {\[Rho]0, -20}}, {{\[Rho]1, 0}, {\[Rho]1, -20}}}]}, 
{Dotted, Red, Line[{{MMaskProj, 0}, {MMaskProj, -20}}]}, 
{Black, Line[{{-1,Log10[contrast]},{50,Log10[contrast]}}]},
{Inset[Style["m/2=" <>ToString[PaddedForm[MMask, {3, 2}]] <>"\!\(\*SubscriptBox[\(\[Lambda]\), \(0\)]\)/\!\(\*SubscriptBox[\(D\), \(Apod\)]\)",Red, FontFamily -> "Helvetica"],  {MMaskProj+ 2.5, -3.}] },
{Inset[Style["\!\(\*SubscriptBox[\(\[Rho]\), \(o\)]\)=" <> ToString[\[Rho]0]<>"\!\(\*SubscriptBox[\(\[Lambda]\), \(0\)]\)/\!\(\*SubscriptBox[\(D\), \(Apod\)]\)",Blue, FontFamily -> "Helvetica"],  {\[Rho]0+2.8, -2.5}] },
{Inset[Style["\!\(\*SubscriptBox[\(\[Rho]\), \(i\)]\)=" <> ToString[PaddedForm[\[Rho]1, {2, 1}]]<>"\!\(\*SubscriptBox[\(\[Lambda]\), \(0\)]\)/\!\(\*SubscriptBox[\(D\), \(Apod\)]\)",Blue, FontFamily -> "Helvetica"],  {\[Rho]1+2.4, -2.5}] },
{Inset[Style[PlotLabelPsfBw, FontFamily -> "Helvetica"],  {40, -11}] }
}]
]


(* ::Subsection::Closed:: *)
(*Pupil plane Array Plot*)


PupilPlaneArrayPlot[PreArrayPlot_]:=Module[{},
Legended[ArrayPlot[PreArrayPlot, ColorFunction -> "SunsetColors", Frame -> False,PlotRange-> {0., 1.}],
 BarLegend[{"SunsetColors", {0., 1.}}, LegendMargins->10,  LegendMarkerSize -> 375, LabelStyle->Directive[FontFamily->"Helvetica", FontSize -> 20.]]
]
]


(* ::Subsection::Closed:: *)
(*Image Plane Array Plot*)


ImagePlaneArrayPlot[PreArrayPlot_]:=Module[{},
ArrayPlot[Log10[PreArrayPlot], ColorFunction -> ColorData["Rainbow"], Frame->False, PlotRange->{-12,-2},
    PlotLegends -> BarLegend[{ColorData[{"Rainbow", "Reverse"}], {-12, -2}},
                              LabelStyle -> Directive[FontFamily->"Helvetica", FontSize -> 20.],
                              LegendMargins->10, LegendMarkerSize -> 375
                              
    ]
]
]


(* ::Subsection::Closed:: *)
(*Fits export*)


stringpad[s_,n_]:=StringJoin[s,Table[" ",{n-StringLength@s}]];
writefitshead[f_][hdat_]:=(BinaryWrite[f,stringpad[StringJoin[stringpad[#[[1]],8],If[Length[#]>1,StringJoin["= ",#[[2]]," "],""],If[Length[#]>2,StringJoin["/",#[[3]]],""]],80]]&/@hdat;
BinaryWrite[f,stringpad["",80]]&/@Range[36-Length[hdat]];)
bytelen["Real64"]=8;
bytelen["Integer16"]=2;
bytelen["Integer8"]=1;
writefitsdat[f_,type_][dat_]:=(BinaryWrite[f,dat,type,ByteOrdering->1];
BinaryWrite[f,Table[0,{i,2880-Mod[Length[dat] bytelen[type],2880]}],"Integer8"]);


FitsExport[filename_,data_]:=Module[{f, header},
header={
{"SIMPLE","T","conforms to FITS standard "},
{"BITPIX","-64","array data type"},
(*{"NAXIS","\"" <>ToString[Depth[data]-1] <>"\""," number of array dimensions"},
{"NAXIS1","\"" <>ToString[Dimensions[data][[1]]] <> "\""},
{"NAXIS2","\"" <>ToString[Dimensions[data][[2]]]<> "\""},*)
{"NAXIS",ToString[Depth[data]-1]," number of array dimensions"},
{"NAXIS1",ToString[Dimensions[data][[1]]]},
{"NAXIS2",ToString[Dimensions[data][[2]]]},
(*,{"EXTEND","T"}*)
{"END"}};

f=OpenWrite[filename,BinaryFormat->True];
writefitshead[f]@header;
writefitsdat[f,"Real64"]@data;(*note the type here has to match BITPIX*)
Close[f];
]


(* ::Section:: *)
(*Parameters*)


(* ::Subsection:: *)
(*Reference*)


(* ::Subsection:: *)
(*Units*)


(* ::Text:: *)
(*Definition of units*)


mm=10.^-3;
\[Mu]m=10.^-6;
nm=10.^-9;


(* ::Subsection:: *)
(*Export*)


(* ::Text:: *)
(*Keyword to export the final apodizer in fits file*)


KeywordExportFitsFiles=0;


(* ::Subsection:: *)
(*Partial pupil*)


(* ::Text:: *)
(*Keyword to define whether optimization was made over quarter pupil or half pupil (Full keyword =1 for robust designs - TBT)*)


QuarterKeyword=0;
FullKeyword=1;


(* ::Text:: *)
(*Parameters from HiCATApertureGeneration - for reference*)


(* ::Text:: *)
(*ApodizerMaskUndersizeContour = 0.972;*)
(*ApodizerMaskOversizeCentralSegmentRatioIrisAO = 0.195;*)
(*SymSpidersKeyword = 1;*)
(*ApodizerMaskSpidersThickRatioIrisAO = 0.017;*)
(*ApodizerMaskGapHexagonRatioIrisAO = 0.004;*)
(**)


(* ::Text:: *)
(*Robust design *)


LyotRobustnessKeyword=1;
MaxShiftStepPix=1.00;
If[LyotRobustnessKeyword == 1, 
FullKeyword =1; 
QuarterKeyword=0;
 ]


(* ::Subsection:: *)
(*Sampling*)


(* ::Text:: *)
(*Pupil size*)


nPup=290;


(* ::Text:: *)
(*FPM sampling (by default, MaskSampling is equal to 100 pixels in ToLyotPlane functions)*)


MaskSampling0=50;


(* ::Text:: *)
(*Maximum spatial frequency in the final image plane (in \[Lambda]0/D0 unit, where D0 denotes the aperture diameter) and sizes for 2D array*)


UndersizedPup=0.807;
nFmax =16.0/UndersizedPup;
nFmax2D=2.*nFmax;


(* ::Text:: *)
(*Array size in the final image plane*)


nImg =32;
nImg2D=2*nImg;


(* ::Subsection:: *)
(*Resized factor*)


(* ::Text:: *)
(*Oversizing factor to increase the size of the pupil arrays*)


ResizingFactor=1;
nPupSuper=ResizingFactor*nPup;


(* ::Text:: *)
(*Keyword to round the values of the apodizer in the absence of oversizing factor*)


RoundApodKeyword=0;


(* ::Subsection:: *)
(*Focal plane mask*)


(* ::Text:: *)
(*Relative mask radius (in \[Lambda]0/D0 unit, where D0 denotes the aperture diameter - do not change the name to Mask, variable is already used in ToLyotPlane functions)*)


(*MMask=6.271/2;*)
MMask=8.543/2;


(* ::Subsection:: *)
(*Dark zone*)


(* ::Text:: *)
(*Dark zone boundaries (\[Rho]0 and \[Rho]1, inner and outer edges in \[Lambda]0/D, where D denotes the pupil diameter), *)


\[Rho]0=3.75;
\[Rho]1=15.0;


(* ::Text:: *)
(*Contrast target for each region*)


cDarkHoleRegion=10.^-8;


(* ::Subsection:: *)
(*Wavelength*)


(* ::Text:: *)
(*Central wavelength \[Lambda]0 (arbitrary unit), bandwidth \[CapitalDelta]\[Lambda] in \[Lambda]0 unit, and wavelength sampling (1- used in the solver, 2- used for the present computation)*)


\[Lambda]0=1.;
bandwidth=0.10;
\[CapitalDelta]\[Lambda]=bandwidth*\[Lambda]0;
n\[Lambda]Solver=3;
n\[Lambda]=3;


(* ::Text:: *)
(*Arrays of wavelengths (1- used in the Optmizer, 2- used for the present computation)*)


If[n\[Lambda]Solver == 1, 
\[Lambda]TableSolver={\[Lambda]0},
\[Lambda]TableSolver=Table[\[Lambda]0*(1+ ((i\[Lambda]-1)/(n\[Lambda]Solver-1.)-0.5)*\[CapitalDelta]\[Lambda]), {i\[Lambda], 1, n\[Lambda]Solver}];]
If[n\[Lambda] == 1, 
\[Lambda]Table={\[Lambda]0},
\[Lambda]Table=Table[\[Lambda]0*(1+ ((i\[Lambda]-1)/(n\[Lambda]-1.)-0.5)*\[CapitalDelta]\[Lambda]), {i\[Lambda], 1, n\[Lambda]}];]


(* ::Text:: *)
(*Wavelength on the HiCAT testbed - for reference*)


(* ::Input:: *)
(*\[Lambda]HiCAT=639.8 nm;*)


(* ::Subsection:: *)
(*Lyot stop*)


(* ::Subsubsection:: *)
(*Parameters from HiCATApertureGeneration.nb - for reference*)


KeywordLyotStopGray = 1;
LyotStopMaskCentralSegmentSizeRatioIrisAO = 0.345;
LyotStopSizeRatioIrisAO = 0.807;
LyotStopMaskSpidersThickRatioIrisAO = 0.036;


(* ::Subsubsection:: *)
(*Dimensions*)


(* ::Text:: *)
(*Projected radius of the FPM in the final image plane (in \[Lambda]0/D unit) *)


MMaskProj=MMask*UndersizedPup


(* ::Subsubsection:: *)
(*Shift versions of the Lyot stop*)


(* ::Text:: *)
(*Resizing of the original array size*)


nPupForLyotStopShift=nPup+20;


(* ::Text:: *)
(*Ratio between old and new sizes of the pupil arrays*)


LyotStopShiftResizingFactor=nPupForLyotStopShift/nPup//N


(* ::Text:: *)
(*Number of shifts, ShiftStep in pixels, and shift direction*)


nShift=5;
ShiftStep=1.00;
YaxisShiftKeyword=1;
XaxisShiftKeyword=1;


(* ::Subtitle:: *)
(*File import and Full pupil generation*)


(* ::Section:: *)
(*File Import*)


(* ::Subsection:: *)
(*Aperture*)


(* ::Subsubsection:: *)
(*Filename and fielpath for Aperture*)


(*FilenameAperturePartial = "HiCAT-Aper_F-N0243_Hex3-Ctr0972-Obs0195-SpX0017-Gap0004.dat"*)
(*FilepathAperturePartial= "input_files/aperture/"<>FilenameAperturePartial;*)
(*FilepathAperture=FileNameJoin[{FileDirectoryHiCATFiles, FilepathAperturePartial}];*)
FilepathAperture = "/Users/kstlaurent/Documents/SCDA_2/input_files/apertures/HiCAT-Aper_F-N0116_Hex3-Ctr0972-Obs0195-SpX0017-Gap0004.fits"


(* ::Subsubsection:: *)
(*Aperture Import*)


(*AperturePartial=Import[FilepathAperture]*)
AperturePartial=Import[FilepathAperture, {"RawData", 1}];
Dimensions[AperturePartial]
ArrayPlot[AperturePartial, PlotRange-> All, ColorFunction->GrayLevel]


(* ::Subsection:: *)
(*Lyot Stop *)


(* ::Subsubsection:: *)
(*Filename and filepath for Lyot Stop*)


(*FilenameLyotStopPartial = "HiCAT-Lyot_F-N0243_LS-Ann-gy-ID0345-OD0807-SpX0036_shiftX+000.dat"
FilepathLyotStopPartial="input_files/lyot_stop/"<>FilenameLyotStopPartial;*)
(*FilepathLyotStop=FileNameJoin[{FileDirectoryHiCATFiles, FilepathLyotStopPartial}];*)
FilepathLyotStop = "/Users/kstlaurent/Documents/SCDA_2/input_files/lyot_stops/HiCAT-Lyot_F-N0116_LS-Ann-gy-ID0345-OD0740-SpX0036.fits"


(* ::Subsubsection:: *)
(*Lyot stop Import*)


LyotStopPartial=Import[FilepathLyotStop,{"RawData",1}];
Dimensions[LyotStopPartial]
ArrayPlot[LyotStopPartial, PlotRange-> All, ColorFunction-> GrayLevel]


(* ::Subsection:: *)
(*Shaped pupil*)


(* ::Subsubsection:: *)
(*Filename and filepath for Shaped Pupil*)


FilepathShapedPupil = "/Users/kstlaurent/Documents/SCDA_2/apodizers/HiCAT_APLC_IWA=3.75_OWA=15.0_BW=0.10_nlam=03_2D_nPup=0116_rMask=4.271_MaxTau_C=8.0_stdgrb.fits"


(* ::Subsubsection:: *)
(*Shaped pupil Import*)


ShapedPupilPartial=Import[FilepathShapedPupil,{"RawData",1}];
(*ShapedPupilDataPartial2=ShapedPupil3DPartial2[[All, 3]];
If[QuarterKeyword ==1,
ShapedPupilPartial2=Partition[ShapedPupilDataPartial2, nPup/2],
ShapedPupilPartial2=Partition[ShapedPupilDataPartial2, nPup]];*)



ArrayPlot[ShapedPupilPartial, ColorFunction -> GrayLevel]


(* ::Section:: *)
(*Full pupil generation*)


(* ::Subsection:: *)
(*Clear aperture*)


(* ::Text:: *)
(*Generation of a clear aperture. Used for the computation of core throughput defined by J. Krist *)


ClearAperture=Cr\[EAcute]eDiaphragmeCenter[nPup/2., nPup];
(*ArrayPlot[ClearAperture,  ColorFunction \[Rule] GrayLevel, PlotRange \[Rule] All, Frame\[Rule] False]*)


(* ::Subsection:: *)
(*Aperture*)


Aperture=If[FullKeyword==1,
AperturePartial,
If[QuarterKeyword==1,
FullPupilFromQuarterGenerationFunction[AperturePartial, nPup],
FullPupilFromHalfGenerationFunction[AperturePartial, nPup]];
];
ArrayPlot[Aperture, ColorFunction -> GrayLevel, Frame-> False]


(* ::Subsection:: *)
(*Lyot stop*)


(* ::Subsubsection:: *)
(*Full Lyot Stop generation*)


LyotStop=If[FullKeyword==1,
LyotStopPartial,
If[QuarterKeyword==1,
FullPupilFromQuarterGenerationFunction[LyotStopPartial, nPup],
FullPupilFromHalfGenerationFunction[LyotStopPartial, nPup]];
];
ArrayPlot[LyotStop, ColorFunction -> "GrayTones", Frame -> False]


(* ::Subsubsection:: *)
(*Generation of shifted Lyot stop*)


(* ::Text:: *)
(*Array of shifted Lyot stops *)


ResizeLyotStopSpiders=Resize[LyotStop, nPupForLyotStopShift];
ResizedShiftedLyotStopArray=Table[0., {nShift},{nPupForLyotStopShift}, {nPupForLyotStopShift}];


Dimensions@ResizeLyotStopSpiders


If[YaxisShiftKeyword==1,
If[XaxisShiftKeyword==0,
Do[
ResizedShiftedLyotStopArray[[iShift, All,All]]=ShiftImageInterp[ResizeLyotStopSpiders, ((nShift +1)/2 -iShift)*ShiftStep, 0.] ;
ResizedShiftedLyotStopArray[[nShift -iShift+1, All,All]]= Reverse[ResizedShiftedLyotStopArray[[iShift, All,All]],1];
,{iShift,1, (nShift+1)/2-1}];
Print["Shift for the Lyot Stop along Y axis"];,
Do[
ResizedShiftedLyotStopArray[[iShift, All,All]]=ShiftImageInterp[ResizeLyotStopSpiders, ((nShift +1)/2 -iShift)*ShiftStep*Sqrt[2.]/2., ((nShift +1)/2 -iShift)*ShiftStep*Sqrt[2.]/2.] ;
ResizedShiftedLyotStopArray[[nShift -iShift+1, All,All]]= Reverse[Reverse[ResizedShiftedLyotStopArray[[iShift, All,All]],1],2];
,{iShift,1, (nShift+1)/2-1}];
Print["Shift for the Lyot Stop along 45deg. axis"];
],
If[XaxisShiftKeyword==1,
Do[
ResizedShiftedLyotStopArray[[iShift, All,All]]=ShiftImageInterp[ResizeLyotStopSpiders, 0.,((nShift +1)/2 -iShift)*ShiftStep];
ResizedShiftedLyotStopArray[[nShift -iShift+1, All,All]]= Reverse[ResizedShiftedLyotStopArray[[iShift, All,All]], 2];
,{iShift,1, (nShift+1)/2-1}];
Print["Shift for the Lyot Stop along X axis"];,
Print["No Shift for the Lyot Stop"];
ResizedShiftedLyotStopArray[[All, All,All]]=ResizeLyotStopSpiders;
]
];
ResizedShiftedLyotStopArray[[(nShift+1)/2, All,All]]=ResizeLyotStopSpiders;


(* ::Text:: *)
(*Array of differences between shifted and centered Lyot stops*)


DifferenceResizedShiftedLyotStopArray=Table[ResizedShiftedLyotStopArray[[iShift]]-ResizedShiftedLyotStopArray[[(nShift+1)/2]], {iShift, 1, nShift}];


(* ::Subsubsection:: *)
(*Display*)


iShift0=1.;
ArrayPlot[ResizedShiftedLyotStopArray[[iShift0]]]
If[nShift >=3,
iShift0=(*5*)nShift;
ArrayPlot[DifferenceResizedShiftedLyotStopArray[[iShift0]], ColorFunction-> GrayLevel, Frame-> False]
]


(* ::Subsection:: *)
(*Shaped pupil*)


(* ::Subsubsection:: *)
(*Full pupil generation*)


(*ShapedPupil=If[FullKeyword==1,
ShapedPupilPartial,
If[QuarterKeyword==1,
FullPupilFromQuarterGenerationFunction[ShapedPupilPartial, nPup],
FullPupilFromHalfGenerationFunction[ShapedPupilPartial, nPup]]
];*)
ShapedPupil = ShapedPupilPartial
ShapedPupilPlot = ArrayPlot[ShapedPupil, ColorFunction -> GrayLevel, Frame -> False]


(* ::Input:: *)
(**)


(* ::Subsubsection:: *)
(*Binarity tests*)


(* ::Text:: *)
(*Comparison between shaped pupil and its rounded version*)


ShapedPupilWithPupil=ShapedPupil*Aperture;
RoundShapedPupilWithPupil=Round[ShapedPupilWithPupil];
DiffShapedPupilWithPupil=Abs[ShapedPupilWithPupil-RoundShapedPupilWithPupil];
ArrayPlot[ShapedPupilWithPupil,  ColorFunction -> "GrayTones", Frame-> False]
ArrayPlot[DiffShapedPupilWithPupil,  ColorFunction -> GrayLevel, Frame -> False]


(* ::Subtitle:: *)
(*Resize with error diffusion algorithm*)


(* ::Subsection:: *)
(*Increase size array*)


(* ::Subsubsection:: *)
(*Pupil size increase for apertures and Lyot stop*)


(* ::Text:: *)
(*Resizing of the different pupil arrays*)


If[ResizingFactor==1,
SuperClearAperture=ClearAperture;
SuperAperture=Aperture;
SuperLyotStop=LyotStop;,
SuperClearAperture=SizeIncrease[ClearAperture, ResizingFactor];
SuperAperture=SizeIncrease[Aperture, ResizingFactor];
SuperLyotStop=SizeIncrease[LyotStop, ResizingFactor];
]


(* ::Subsubsection:: *)
(*Pupil size increase for the apodizer*)


(* ::Text:: *)
(*Resizing of the shaped pupil array and possible use of error diffusion algorithm*)


If[ResizingFactor==1,
SuperShapedPupil=If[RoundApodKeyword==1,Round[ShapedPupil],ShapedPupil];,
SuperShapedPupil=SizeIncreaseAndErrorDiffusion[ShapedPupil, ResizingFactor];
]


(* ::Subsubsection:: *)
(*Pupil size increase for the shifted Lyot stops*)


(* ::Text:: *)
(*Resizing of the shifted Lyot stop*)


If[ResizingFactor==1,
SuperResizedShiftedLyotStopArray=ResizedShiftedLyotStopArray;,
SuperResizedShiftedLyotStopArray=Table[0., {nShift}];
Do[
SuperResizedShiftedLyotStopArray[[iShift]]=SizeIncrease[ResizedShiftedLyotStopArray[[iShift]], ResizingFactor];,
{iShift,1, nShift}];
]


(* ::Subsection:: *)
(*Clear All the original arrays*)


If[ResizingFactor!= 1 && KeywordExportFitsFiles != 1,
ClearAll[ClearAperture];
(*ClearAll[Aperture]*);
(*ClearAll[LyotStop]*);
(*ClearAll[ShapedPupil]*);
ClearAll[ResizedShiftedLyotStopArray];
]


(* ::Subsection:: *)
(*Oversized pupils*)


If[nPupForLyotStopShift != nPup,
nPupHyper=nPupForLyotStopShift*ResizingFactor;
HyperClearAperture=Resize[SuperClearAperture, nPupHyper];
HyperAperture=Resize[SuperAperture, nPupHyper];
HyperShapedPupil=Resize[SuperShapedPupil, nPupHyper];,
nPupHyper=nPupForLyotStopShift;
HyperClearAperture=SuperClearAperture;
HyperAperture=SuperAperture;
HyperShapedPupil=SuperShapedPupil;
]


MemoryInUse[]


(* ::Subtitle:: *)
(*APLC SP propagation*)


(* ::Section:: *)
(*Lyot plane*)


(* ::Subsection:: *)
(*Lyot plane field, before stop application*)


BeforeLyotFieldLP2D0All=Table[0., {i\[Lambda], 1, n\[Lambda]}];
BeforeLyotFieldLP2DAll=Table[0., {i\[Lambda], 1, n\[Lambda]}];

Do[
\[Lambda]=\[Lambda]Table[[i\[Lambda]]];
BeforeLyotFieldLP2D0All[[i\[Lambda]]]=ToLyotPlaneCenter[HyperShapedPupil*HyperAperture, 0., Wavelength-> \[Lambda]/\[Lambda]0, OverSizePupil->LyotStopShiftResizingFactor, MaskSampling -> MaskSampling0];
BeforeLyotFieldLP2DAll[[i\[Lambda]]]=ToLyotPlaneCenter[HyperShapedPupil*HyperAperture, 2.*MMask, Wavelength-> \[Lambda]/\[Lambda]0, OverSizePupil->LyotStopShiftResizingFactor, MaskSampling -> MaskSampling0];
,{i\[Lambda], 1, n\[Lambda]}];



MemoryInUse[]


(* ::Subsection:: *)
(*Lyot plane field, after stop application*)


LyotFieldLP2D0AllTable=Table[0., {iShift, 1, nShift}];
LyotFieldLP2DAllTable=Table[0., {iShift, 1, nShift}];

Do[
LyotFieldLP2D0All=Table[0., {i\[Lambda], 1, n\[Lambda]}];
LyotFieldLP2DAll=Table[0., {i\[Lambda], 1, n\[Lambda]}];

Do[
LyotFieldLP2D0All[[i\[Lambda]]]=BeforeLyotFieldLP2D0All[[i\[Lambda]]]*SuperResizedShiftedLyotStopArray[[iShift]];
LyotFieldLP2DAll[[i\[Lambda]]]=BeforeLyotFieldLP2DAll[[i\[Lambda]]]*SuperResizedShiftedLyotStopArray[[iShift]];


,{i\[Lambda], 1, n\[Lambda]}];
LyotFieldLP2D0AllTable[[iShift]]=LyotFieldLP2D0All;
LyotFieldLP2DAllTable[[iShift]]=LyotFieldLP2DAll;

, {iShift, 1, nShift}]


If[ResizingFactor!= 1,
Clear[BeforeLyotFieldLP2D0All];
Clear[BeforeLyotFieldLP2DAll];
Clear[LyotFieldLP2D0All];
Clear[LyotFieldLP2DAll];
]


MemoryInUse[]


Dimensions[BeforeLyotFieldLP2DAll]
Dimensions[LyotFieldLP2DAllTable]


(* ::Subsection:: *)
(*Display*)


(* ::Subsubsection:: *)
(*Without DM*)


i\[Lambda]0=(n\[Lambda]+1)/2;
If[ResizingFactor==  1,
PreArrayPlot=Abs[BeforeLyotFieldLP2DAll[[i\[Lambda]0,All,All]]]^0.5;
(*PupilPlaneArrayPlot[PreArrayPlot]*)
]



i\[Lambda]0=(n\[Lambda]+1)/2;
If[ResizingFactor==  1,
PreArrayPlot=Abs[LyotFieldLP2DAllTable[[(nShift+1)/2, i\[Lambda]0,All,All]]]^0.5;
(*PupilPlaneArrayPlot[PreArrayPlot]*)
]


If[ResizingFactor==  1,
If[nShift > 1,
PreArrayPlot=Abs[LyotFieldLP2DAllTable[[(nShift+1)/2+1, (n\[Lambda]+1)/2,All,All]]]^0.5;
(*PupilPlaneArrayPlot[PreArrayPlot]*)
]
]


If[ResizingFactor==  1,
If[nShift > 2,
PreArrayPlot=Abs[LyotFieldLP2DAllTable[[(nShift+1)/2+2, (n\[Lambda]+1)/2,All,All]]]^0.5;
(*PupilPlaneArrayPlot[PreArrayPlot]*)
]
]


If[ResizingFactor==  1,
If[nShift > 3,
PreArrayPlot=Abs[LyotFieldLP2DAllTable[[(nShift+1)/2, (n\[Lambda]+1)/2,All,All]]]^0.5;
(*PupilPlaneArrayPlot[PreArrayPlot]*)
]
]


Dimensions[LyotFieldLP2DAllTable]


(* ::Section:: *)
(*Camera plane*)


(* ::Subsection:: *)
(*Final image plane field*)


FinalFieldLP2D0AllTable=Table[0., {iShift, 1, nShift}];
FinalFieldLP2DAllTable=Table[0., {iShift, 1, nShift}];


Do[
FinalFieldLP2D0All=Table[0., {i\[Lambda], 1, n\[Lambda]}];
FinalFieldLP2DAll=Table[0., {i\[Lambda], 1, n\[Lambda]}];

Do[
\[Lambda]=\[Lambda]Table[[i\[Lambda]]];
FinalFieldLP2D0All[[i\[Lambda]]]=(\[Lambda]0/\[Lambda])*SFT[LyotFieldLP2D0AllTable[[iShift]][[i\[Lambda]]], nFmax2D*(\[Lambda]0/\[Lambda])*UndersizedPup*LyotStopShiftResizingFactor,nImg2D, Centering -> "Pixel"];
FinalFieldLP2DAll[[i\[Lambda]]]=(\[Lambda]0/\[Lambda])*SFT[LyotFieldLP2DAllTable[[iShift]][[i\[Lambda]]], nFmax2D*(\[Lambda]0/\[Lambda])*UndersizedPup*LyotStopShiftResizingFactor,nImg2D, Centering -> "Pixel"];


,{i\[Lambda], 1, n\[Lambda]}];

FinalFieldLP2D0AllTable[[ iShift]]=FinalFieldLP2D0All;
FinalFieldLP2DAllTable[[ iShift]]=FinalFieldLP2DAll;

, {iShift, 1, nShift}]


If[ResizingFactor !=  1,
Clear[LyotFieldLP2D0AllTable];
Clear[LyotFieldLP2DAllTable];
Clear[FinalFieldLP2D0All];
Clear[FinalFieldLP2DAll];
]


(* ::Subsection:: *)
(*Computation of images*)


(* ::Subsubsection:: *)
(*Monochromatic images*)


FinalImageLP2D0AllTable=Abs[FinalFieldLP2D0AllTable]^2;
FinalCoronoLP2DAllTable=Abs[FinalFieldLP2DAllTable]^2;


Dimensions[FinalCoronoLP2DAllTable]


MemoryInUse[]


If[ResizingFactor !=  1,
Clear[FinalFieldLP2D0AllTable];
Clear[FinalFieldLP2DAllTable];
]


(* ::Subsubsection:: *)
(*Broadband images*)


PolyImagePSFLP2D0Table=Total[FinalImageLP2D0AllTable, {2}];
PolyCoronoPSFLP2DTable=Total[FinalCoronoLP2DAllTable, {2}];


(* ::Input:: *)
(*Dimensions[PolyCoronoPSFLP2DTable]*)


(* ::Subsubsection:: *)
(*Normalization for monochromatic images*)


MonoCoronoImageArray=Table[0., {nShift}];


Do[MonoCoronoImageArray[[iShift ]]=FinalCoronoLP2DAllTable[[iShift, (n\[Lambda]+1)/2]]/FinalImageLP2D0AllTable[[iShift,(n\[Lambda]+1)/2, nImg2D/2+1, nImg2D/2+1]];
, {iShift, 1, nShift}]


(* ::Subsubsection:: *)
(*Normalization for broadband images*)


PolyImagePSFLP2D0NTable=Table[0., {nShift}];
PolyImagePSFLP2DCANTable=Table[0., {nShift}];
PolyImagePSFLP2D00NTable=Table[0.,{nShift}];
PolyCoronoPSFLP2DNTable=Table[0., {nShift}];


Do[
PolyImagePSFLP2D0NTable[[iShift]]=PolyImagePSFLP2D0Table[[iShift]]/PolyImagePSFLP2D0Table[[iShift, nImg2D/2+1, nImg2D/2+1]];
PolyCoronoPSFLP2DNTable[[iShift]]=PolyCoronoPSFLP2DTable[[iShift]]/PolyImagePSFLP2D0Table[[iShift, nImg2D/2+1, nImg2D/2+1]];
, {iShift, 1, nShift}];


Dimensions[PolyCoronoPSFLP2DNTable]


(* ::Subsection:: *)
(*Display - Camera image in monochromatic light*)


(* ::Subsubsection:: *)
(*Performance without DM*)


(* ::Text:: *)
(*Coronagraphic image*)


iShift0=(nShift+1)/2;
ImagePlaneArrayPlot[MonoCoronoImageArray[[iShift0]]]


(* ::Input:: *)
(*(*Max[PolyImagePSFLP2D0NTable]*)*)


(* ::Subsection:: *)
(*Display - Camera image in broadband light*)


(* ::Subsubsection:: *)
(*Performance without DM*)


(* ::Text:: *)
(*Coronagraphic image*)


iShift0=(nShift+1)/2+2;
BroadbandPSF = ImagePlaneArrayPlot[PolyCoronoPSFLP2DNTable[[iShift0]]]


(* ::Subtitle:: *)
(*1D radial cut profiles*)


(* ::Subsection:: *)
(*Scale parameters*)


Xi = Table[k*nFmax/nImg*UndersizedPup,{k,0,nImg}];
(*Xi2D=Table[nFmax2D/nImg2D(k-(nImg2D/2+1))*UndersizedPup,{k,1,nImg2D}];*)
Xi2D=Table[nFmax2D/nImg2D (k-nImg2D/2)*UndersizedPup,{k,1,nImg2D}];
Xi2DinDlyot=Table[nFmax2D/nImg2D (k-nImg2D/2)*UndersizedPup*UndersizedPup,{k,1,nImg2D}];(*THIS IS A HACK THE FORMULA IS WRONG WE MUST HAVE A MULTIPLY vs. DIVIDE ERROR SOMEWHERE*)


PlotLabelPsfObs= (*If[IRISAOshape \[Equal] 1,"IRISAO, 15% central obstruction",*) (*ToString[Round[ratio*100]] <>*) "xx% central obstruction"(*]*);
PlotLabelPsfMono="";
PlotLabelPsfBw= ToString[Round[bandwidth*100]]<> "% bandwidth" ;
PlotLabelPsfMask="Mask radius m/2="  <>  ToString[MMask] <> "\!\(\*SubscriptBox[\(\[Lambda]\), \(0\)]\)/D" ;


(* ::Subsection:: *)
(*1D radial cut coronagraphic profile at Subscript[\[Lambda], 0] for different Lyot stop decentering -> monochrom*)


(* ::Subsubsection:: *)
(*Without DM*)


(* ::Text:: *)
(*Coronagraphic images*)


iShift0=(nShift+1)/2;
PreListPlot1=
Table[Table[{Xi2D[[k2D]],Log10[FinalCoronoLP2DAllTable[[iShift,(n\[Lambda]+1)/2,k2D, nImg2D/2+1]]/FinalImageLP2D0AllTable[[1,(n\[Lambda]+1)/2,nImg2D/2+1, nImg2D/2+1]]]}, {k2D, 1, nImg2D}],
{iShift, 1, nShift}];
Label1 = "1D radial cut coronagraphic profile at \!\(\*SubscriptBox[\(\[Lambda]\), \(0\)]\) for different Lyot Stop decenterings";


ContrastListPlot[PreListPlot1,MMask,\[Rho]0,\[Rho]1,cDarkHoleRegion,Label1]


(* ::Subsection:: *)
(*1D radial cut profiles for different Lyot stop decentering in broadband light*)


(* ::Subsubsection:: *)
(*without DM*)


(* ::Text:: *)
(*Coronagraphic image*)


iShift0=(nShift+1)/2;
PreListPlot2=Table[Table[{Xi2D[[k2D]],Log10[PolyCoronoPSFLP2DNTable[[iShift, k2D, nImg2D/2+1]]]}, {k2D, 1, nImg2D}], {iShift, 1, nShift}];
Label2 = "1D radial cut profiles for different Lytot stop decententerings in broadband light";


BroadbandMultiLSRadialCut=ContrastListPlot[PreListPlot2,MMask,\[Rho]0,\[Rho]1,cDarkHoleRegion,Label2]


(* ::Subsection:: *)
(*2D radial cut profiles at all the wavelengths*)


(* ::Subsubsection:: *)
(*Without DM*)


PreListPlot3=Table[Table[{Xi2D[[k2D]],Log10[FinalCoronoLP2DAllTable[[iShift0, i\[Lambda], nImg2D/2+1, k2D]]/FinalImageLP2D0AllTable[[iShift0, (n\[Lambda]+1)/2,nImg2D/2+1, nImg2D/2+1]]]}, {k2D, 1, nImg2D}](*, {i\[Lambda], {1, (1 + n\[Lambda])/2, n\[Lambda]}}*), {i\[Lambda], 1, n\[Lambda]}];
Label3 = "2D radial cut profiles at all wavelengths";
BroadbandRadialCut=ContrastListPlot[PreListPlot3,MMask,\[Rho]0,\[Rho]1,cDarkHoleRegion,Label3]


(* ::Subtitle:: *)
(*Averaged intensity profiles*)


(* ::Subsection:: *)
(*Monochromatic light *)


(* ::Subsubsection::Closed:: *)
(*Azimuthal averaged intensity profiles in monochromatic light *)


AziMonoImagePSFLP2D0NTable=Table[0,{iShift, 1, nShift}];
AziMonoCoronoPSFLP2DNTable=Table[0,{iShift, 1, nShift}];
NormAziMonoImagePSFLP2D0NTable=Table[0, {iShift, 1, nShift}];
NormAziMonoCoronoPSFLP2DNTable=Table[0, {iShift, 1, nShift}];




Do[
AziMonoImagePSFLP2D0NTable[[iShift]] = AzimuthalAverage[FinalImageLP2D0AllTable[[iShift,(n\[Lambda]+1)/2, All, All]]];AziMonoCoronoPSFLP2DNTable[[iShift]] = AzimuthalAverage[FinalCoronoLP2DAllTable[[iShift, (n\[Lambda]+1)/2,All, All]]];
NormAziMonoImagePSFLP2D0NTable[[iShift]]=AziMonoImagePSFLP2D0NTable[[iShift,All]]/Max[AziMonoImagePSFLP2D0NTable[[1]]];
NormAziMonoCoronoPSFLP2DNTable[[iShift]]=AziMonoCoronoPSFLP2DNTable[[iShift,All]]/Max[AziMonoImagePSFLP2D0NTable[[1]]];
    ,{iShift, 1, nShift}]


(* ::Subsubsection:: *)
(*Display - 1D azimuth averaged profile in monochromatic light*)


iShift0=(nShift+1)/2;


PreListPlot4=Table[Table[{Xi[[k1D]],Log10[NormAziMonoCoronoPSFLP2DNTable[[iShift,k1D]]]}, {k1D, 1, nImg}], {iShift, 1, nShift}];
Label4 = "1D azimuth averaged profile in monochromatic light";
ContrastListPlot[PreListPlot4,MMask,\[Rho]0,\[Rho]1,cDarkHoleRegion,Label4]


(* ::Subsection:: *)
(*Broadband light*)


(* ::Subsubsection::Closed:: *)
(*Azimuthal averaged intensity profiles in broadband light *)


AziPolyImagePSFLP2D0NTable=Table[0,{iShift, 1, nShift}];
AziPolyCoronoPSFLP2DNTable=Table[0,{iShift, 1, nShift}];
NormAziPolyCoronoPSFLP2DNTable=Table[0, {iShift, 1, nShift}];

Do[
AziPolyImagePSFLP2D0NTable[[iShift]] = AzimuthalAverage[PolyImagePSFLP2D0NTable[[iShift, All, All]]];
AziPolyCoronoPSFLP2DNTable[[iShift]] = AzimuthalAverage[PolyCoronoPSFLP2DNTable[[iShift, All, All]]];
NormAziPolyCoronoPSFLP2DNTable[[iShift]]=AziPolyCoronoPSFLP2DNTable[[iShift,All]]/Max[AziPolyImagePSFLP2D0NTable[[1]]];
    ,{iShift, 1, nShift}]


(* ::Subsubsection:: *)
(*1D azimuth averaged profile in broadband light for all the Lyot stop shifts*)


(* ::Text:: *)
(*Without DM*)


PreListPlot5=Table[Table[{Xi[[k1D]],Log10[NormAziPolyCoronoPSFLP2DNTable[[iShift,k1D]]]}, {k1D, 1, nImg}], {iShift, 1, nShift}];
Label5 = "1D azimuth averaged profile in broadband light for all Lyot stop shifts";
ContrastListPlot[PreListPlot5,MMask,\[Rho]0,\[Rho]1,cDarkHoleRegion,Label5]


(* ::Subtitle:: *)
(*Contrast vs Lyot stop shift*)


\[Rho]2D=nFmax2D/nImg2D*UndersizedPup*Table[Sqrt[(i-nImg2D/2-1)^2+(j-nImg2D/2-1)^2],{i,1,nImg2D},{j,1,nImg2D} ];


DarkZonePts=Position[\[Rho]2D, _?((# >=  \[Rho]0)  && (# <=  \[Rho]1) &)];
DarkZoneArray=Table[0., {nImg2D},{nImg2D} ];
Do[
DarkZoneArray[[DarkZonePts[[i,1]],DarkZonePts[[i,2]]]]=1.;
,{i,1,Length[DarkZonePts]}];



AveragedDZContrast=Table[0., {nShift}];
Do[
AveragedDZContrast[[iShift]]=Total[Log10[MonoCoronoImageArray[[iShift]]]*DarkZoneArray, 2]/Length[DarkZonePts];
, {iShift, 1, nShift}]


ShiftArray=Table[100.*(iShift - (nShift +1)/2)*ShiftStep/(UndersizedPup*nPup), {iShift, 1, nShift}];


Max[ShiftArray]


PreListPlot=Table[{ShiftArray[[iShift]], AveragedDZContrast[[iShift]]}, {iShift, 1, nShift}];
ListPlot[PreListPlot,Joined->True, PlotRange-> {{-1.1,1.1}, {-11, -5}}, ImageSize-> Large,LabelStyle->Directive[FontFamily->"Helvetica", FontSize-> 15.], 
 FrameLabel-> {"Shift in % Lyot stop size", "Averaged intensity in log scale"},
FrameStyle-> AbsoluteThickness[1.1],
PlotStyle->If[Length[PreListPlot]>1, Table[Directive[ColorData["Rainbow",i/(Length[PreListPlot]-1)], AbsoluteThickness[2.0]],{i,0,Length[PreListPlot]-1}], Directive[ColorData["Rainbow",1], AbsoluteThickness[2.0]]],
Epilog ->{FontSize -> 15.,AbsoluteThickness[2.0],
{Black, Line[{{-10, Log10[cDarkHoleRegion]}, {10, Log10[cDarkHoleRegion]}}]}
}]


(* ::Subtitle:: *)
(*Throughput*)


(* ::Section:: *)
(*Total throughput*)


(* ::Text:: *)
(*Be careful, total throughput is different from Airy throughput*)


ApodizerThroughput=100.*(Total[ShapedPupil^2*Aperture^2, 2])/(Total[Aperture^2, 2]);
CoronagraphThroughput=100.*(Total[ShapedPupil^2*Aperture^2*LyotStop^2, 2])/(Total[Aperture^2, 2]);
"Apodizer throughput: " <> ToString[ApodizerThroughput] <> "%"
"Coronagraph throughput: " <> ToString[CoronagraphThroughput] <> "%"


(* ::Subtitle:: *)
(*Export*)


(* ::Section:: *)
(*Fits file*)


(* ::Subsection:: *)
(*No oversize*)


(* ::Subsubsection:: *)
(*Filenames*)


(* ::Text:: *)
(*Aperture*)


FitsFilenameAperture = StringReplace[FilepathAperturePartial, { ".dat" -> ".fits", "Aper_Q" -> "Aper_F", "N" <> PaddingForFileNames[nPup/2, 4] -> "N" <> PaddingForFileNames[nPup, 4]}]
FitsFilepathAperture = StringReplace[FileNameJoin[{FileDirectoryHiCATFiles, FitsFilenameAperture}], {"/dat/" ->  "/fits/"}]


(* ::Text:: *)
(*Lyot stop*)


FitsFilenameLyotStop=StringReplace[FilepathLyotStopPartial,{ ".dat" -> ".fits", "Aper_Q" -> "Aper_F", "N"<> PaddingForFileNames[nPup/2, 4] -> "N"<> PaddingForFileNames[nPup, 4]}]
FitsFilepathLyotStop=StringReplace[FileNameJoin[{FileDirectoryHiCATFiles, FitsFilenameLyotStop}], {"/dat/" ->  "/fits/"}]


(* ::Text:: *)
(*Shaped pupil*)


FitsFilenameShapedPupil=StringReplace[FilepathShapedPupilPartial,{ ".dat" -> ".fits", "Apod_Q" -> "Apod_F", "N"<> PaddingForFileNames[nPup/2, 4] -> "N"<> PaddingForFileNames[nPup, 4]}]
FitsFilepathShapedPupil=StringReplace[FileNameJoin[{FileDirectoryHiCATFiles, FitsFilenameShapedPupil}], {"/dat/" ->  "/fits/"}]


(* ::Subsubsection:: *)
(*Export*)


If[KeywordExportFitsFiles ==1,
FitsExport[FitsFilepathAperture, Aperture];
FitsExport[FitsFilepathShapedPupil, ShapedPupilWithPupil];
FitsExport[FitsFilepathLyotStop, LyotStop];
]


(* ::Subsection:: *)
(*Oversize*)


(* ::Subsubsection:: *)
(*Filenames*)


(* ::Text:: *)
(*Aperture*)


FitsFilenameSuperAperture=StringReplace[FitsFilenameAperture, {"N" <> PaddingForFileNames[nPup, 4] -> "N" <> PaddingForFileNames[nPupSuper, 4], ".fits" -> "_EDA"<> PaddingForFileNames[ResizingFactor, 2]<> ".fits"}]
FitsFilepathSuperAperture=StringReplace[FileNameJoin[{FileDirectoryHiCATFiles, FitsFilenameSuperAperture}], {"/dat/" ->  "/fits/"}]


(* ::Text:: *)
(*Shaped Pupil*)


FitsFilenameSuperShapedPupil=StringReplace[FitsFilenameShapedPupil, {"N" <> PaddingForFileNames[nPup, 4] -> "N" <> PaddingForFileNames[nPupSuper, 4], ".fits" -> "_EDA"<> PaddingForFileNames[ResizingFactor, 2]<> ".fits"}]
FitsFilepathSuperShapedPupil=StringReplace[FileNameJoin[{FileDirectoryHiCATFiles, FitsFilenameSuperShapedPupil}], {"/dat/" ->  "/fits/"}]


(* ::Text:: *)
(*Lyot stop*)


FitsFilenameSuperLyotStop=StringReplace[FitsFilenameLyotStop, {"N" <> PaddingForFileNames[nPup, 4] -> "N" <> PaddingForFileNames[nPupSuper, 4], ".fits" -> "_EDA"<> PaddingForFileNames[ResizingFactor, 2]<> ".fits"}]
FitsFilepathSuperLyotStop=StringReplace[FileNameJoin[{FileDirectoryHiCATFiles, FitsFilenameSuperLyotStop}], {"/dat/" ->  "/fits/"}]


(* ::Subsubsection:: *)
(*Export*)


If[KeywordExportFitsFiles ==1,
If[ResizingFactor!= 1, 
FitsExport[FitsFilepathSuperAperture, SuperAperture];
FitsExport[FitsFilepathSuperShapedPupil, SuperShapedPupil*SuperAperture];
FitsExport[FitsFilepathSuperLyotStop, SuperLyotStop];
]
]


(* ::Subsection:: *)
(*Plots*)


(* ::Subsubsection:: *)
(*Filenames*)


FilenameBroadbandPSF = StringReplace[FitsFilenameSuperShapedPupil,{".fits" -> "_PSF.pdf","output_apodizers/dat/"->"plots/"}]
FilepathBroadbandPSF = FileNameJoin[{FileDirectoryHiCATFiles, FilenameBroadbandPSF}]


FilenameBroadbandMultiLSRadialCut = StringReplace[FitsFilenameSuperShapedPupil,{".fits" -> "_MultiLSRadialProfile.pdf","output_apodizers/dat/"->"plots/"}]
FilepathBroadbandMultiLSRadialCut = FileNameJoin[{FileDirectoryHiCATFiles, FilenameBroadbandMultiLSRadialCut}]


FilenameBroadbandRadialCut = StringReplace[FitsFilenameSuperShapedPupil,{".fits" -> "_RadialProfile.pdf","output_apodizers/dat/"->"plots/"}]
FilepathBroadbandRadialCut = FileNameJoin[{FileDirectoryHiCATFiles, FilenameBroadbandRadialCut}]


PDFFilenameShapedPupil=StringReplace[FilepathShapedPupilPartial,{ ".dat" -> ".pdf", "Apod_Q" -> "Apod_F", "N"<> PaddingForFileNames[nPup/2, 4] -> "N"<> PaddingForFileNames[nPup, 4]}]
PDFFilepathShapedPupil=StringReplace[FileNameJoin[{FileDirectoryHiCATFiles, PDFFilenameShapedPupil}], {"/dat/" ->  "/pdf/"}]


(* ::Subsubsection:: *)
(*Export*)


If[KeywordExportFitsFiles ==1,
Export[FilepathBroadbandPSF, BroadbandPSF];
Export[FilepathBroadbandMultiLSRadialCut,BroadbandMultiLSRadialCut ];
Export[FilepathBroadbandRadialCut,BroadbandRadialCut ];
Export[PDFFilepathShapedPupil,ShapedPupilPlot];
]
