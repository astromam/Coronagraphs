#################################################################
### AMPL script to design APLC/SP hybrid coronagraph with quarter pupil for HiCAT
### Written by Mamadou N'Diaye on 2016/09/21
### v19 - 2017/02/17: addition of constraints on the non coronagraphic PSF
###					  modification of the apodizer name to account for the constraints on the non coronagraphic PSF
###					  Lyot stop robustness constraints for the non coronagraphic PSF are currently set in comments
### Secondary Zone removed
#################################################################
### parameters set by the user
#################################################################
#---------------------
### directory definition
param directoryAperture symbolic;
param directoryLyot symbolic;
param directoryFPM symbolic;
param directoryOutput symbolic;
let directoryAperture := "/user/kstlaurent/SCDA_2/input_files/apertures/";
let directoryLyot := "/user/kstlaurent/SCDA_2/input_files/lyot_stops/";
let directoryFPM := "/user/kstlaurent/SCDA_2/input_files/fpm/";
let directoryOutput := "/user/kstlaurent/SCDA_2/apodizers/HiCAT/";

#---------------------
### pi defintion
param pi := 4*atan(1);

#---------------------
### spectral bandwidth (lam0: arbitrary unit, bw: bandwidth in % of lam0, dlam: bandwidth in arbitrary unit, nlam: spectral sampling)
### optimization in monochromatic light is done by setting bw=0 (and not nlam=1)
param lam0 := 1.;
param bw := 0.10;
param dlam := bw*lam0;
param nlam := 3;

#---------------------
# entrance pupil parameter (use the values showed in the filename generated with HiCATApertureGeneration_vXX.nb) - see also HiCAT_APLCSP_spreadsheet - All three (script/notebook/excel) should be consistent end pushed together to GitHub
param AperType symbolic; let AperType := "Hex3";
param ApodizerMaskUndersizeContour := 0.972;
param ApodizerMaskOversizeCentralSegmentRatioIrisAO := 0.195;
param SpidersType symbolic; let SpidersType := "X";# "X" or "Y"
param ApodizerMaskSpidersThickRatioIrisAO := 0.017;
param ApodizerMaskGapHexagonRatioIrisAO := 0.004;

#---------------------
### FPM radius in lam0/D with D_apodizer
param rMask := 8.543/2;

#---------------------
# Lyot stop parameter  (use the values showed in the filename generated with HiCATApertureGeneration_vXX.nb)
param LyotStopType symbolic; let LyotStopType := "Ann";
param LyotStopBwGy symbolic; let LyotStopBwGy := "bw";# "gy" or "bw"
param LyotStopMaskCentralSegmentSizeRatioIrisAO := 0.345;
param LyotStopSizeRatioIrisAO := 0.807;
param LyotStopMaskSpidersThickRatioIrisAO := 0.036;

param LyotStopShiftPix := 1.0; #for the robustness

#---------------------
### diamater ratio between aperture and Lyot stop
param CoeffOverSizePup := LyotStopSizeRatioIrisAO; 

#---------------------
### Outer diameter of each region of the coronagraphic image in lam0/D_apod
param rho0 := 5.00;
param rho1 := 10.00; # controllable region by the DM is 14.87 lambda/D_apod

#---------------------
### contrast definition in the different regions of the coronagraphic image
param cCoreRegion := 3;
param cDarkHoleRegion := 8;

#---------------------
### discretization parameter (nPup: pupil, nFPM: mask, nImg: image, Fmax: max spatial frequency in image plane)
### be careful, sampling is given for half mask
param nPup := 100;
param nFPM := 25;

#for the coronagraph image
param nImg := 32;
param Fmax := 16; 

#################################################################
### filename definitions
#################################################################
#---------------------
### files for import
param filenameAperture symbolic;
param filenameFPM symbolic;
param filenameLyotStop0 symbolic;
param filenameLyotStop1 symbolic;
param filenameLyotStop2 symbolic;

let filenameAperture := "HiCAT-Aper_F-N0200_Hex3-Ctr0972-Obs0195-SpX0017-Gap0004.dat";

let filenameFPM := "CircPupil_N=00" & nFPM & "_obs=00_center_quarter.dat";

let filenameLyotStop0 := "HiCAT-Lyot_F-N0200_LS-Ann-bw-ID0345-OD0807-SpX0036_shiftX+000.dat";
let filenameLyotStop1 := "HiCAT-Lyot_F-N0200_LS-Ann-bw-ID0345-OD0807-SpX0036_shiftX+100.dat";
let filenameLyotStop2 := "HiCAT-Lyot_F-N0200_LS-Ann-bw-ID0345-OD0807-SpX0036_shiftY+100.dat";

#---------------------
### files for export
param filenameApodizer symbolic;
param filenameApodizerDat symbolic;
param filenameApodizerLog symbolic;


let filenameApodizer := "Test_009_HiCAT-Apod_F-N" & sprintf("%.4d", nPup) & "_nImg" & sprintf("%.4d", nImg) & "_" & sprintf("%s", AperType) & "-Ctr" & sprintf("%.4d", ApodizerMaskUndersizeContour*1000) & "-Obs" & sprintf("%.4d", ApodizerMaskOversizeCentralSegmentRatioIrisAO*1000) & "-Sp" & sprintf("%s%.4d", SpidersType, ApodizerMaskSpidersThickRatioIrisAO*1000) & "-Gap" & sprintf("%.4d", ApodizerMaskGapHexagonRatioIrisAO*1000) & "_GreyFPM" & sprintf("%.4d", rMask*2*1000) & "-M" & sprintf("%.3d", nFPM) & "_LS-" & sprintf("%s", LyotStopType) & "-" & sprintf("%s", LyotStopBwGy) & "-ID" & sprintf("%.4d", LyotStopMaskCentralSegmentSizeRatioIrisAO*1000) & "-OD" & sprintf("%.4d", LyotStopSizeRatioIrisAO*1000) & "-Sp" & sprintf("%s%.4d", SpidersType, LyotStopMaskSpidersThickRatioIrisAO*1000) & "_DZ-C" & sprintf("%.3d", cDarkHoleRegion*10) & "-Sep" & sprintf("%.3d", rho0*10) & "-" & sprintf("%.3d", rho1*10) & "_Bw" & sprintf("%.2d", bw*100) & "-Lam" & nlam & "_shiftXY" & sprintf("%.3d", LyotStopShiftPix*100);
let filenameApodizerDat := sprintf("%s", filenameApodizer) & ".dat" ;


#################################################################
### vector definitions
#################################################################
#---------------------
### array of wavelengths
set Ls := setof {l in 1..nlam} lam0*(1+((l-1)/(nlam-1)-0.5)*dlam);

#---------------------
### discretization in each plane ({dx,dy}: pupil planes, {dmx, dmy}: first focal plane, {dxi, deta}: final image plane)
param dx := 1/(2*nPup);
param dy := dx;

param dmx := 2.*rMask/(2*nFPM);
param dmy := dmx;

param dxi := (Fmax/nImg)*(1/CoeffOverSizePup);
#param dxi := (Fmax/nImg);
param deta := dxi;

#---------------------
### coordinate vectors for each plane ({Xs,Ys}: pupil planes, {MXs, MYs}: first focal plane, {Xis, Etas}: final image plane)
set XsQ := setof {i in 0.5..nPup-0.5 by 1} i*dx;
set YsQ := setof {j in 0.5..nPup-0.5 by 1} j*dy;

set Xs := setof {i in -nPup+0.5..nPup-0.5 by 1} i*dx;
set Ys := setof {j in -nPup+0.5..nPup-0.5 by 1} j*dy;

set MXsQ := setof {i in 0.5..nFPM-0.5 by 1} i*dmx;
set MYsQ := setof {j in 0.5..nFPM-0.5 by 1} j*dmy;

set MXs := setof {i in -nFPM+0.5..nFPM-0.5 by 1} i*dmx;
set MYs := setof {j in -nFPM+0.5..nFPM-0.5 by 1} j*dmy;

set Xis := setof {i in -nImg..nImg-1 by 1} i*dxi;
set Etas := setof {j in -nImg..nImg-1 by 1} j*deta;

#---------------------
### set of points in the final image plane for the coronagraphic image, only for 360deg masks.
set CoroCoreRegion := setof {xi in Xis, eta in Etas: sqrt(xi^2+eta^2) >= 0 && sqrt(xi^2+eta^2) < rho0} (xi,eta);
set CoroDarkHoleRegion := setof {xi in Xis, eta in Etas: sqrt(xi^2+eta^2) >= rho0 && sqrt(xi^2+eta^2) <= rho1} (xi,eta);

set CoroOuterRegionAll := setof {xi in Xis, eta in Etas: sqrt(xi^2+eta^2) >= 0 && sqrt(xi^2+eta^2) <= rho1} (xi,eta);

#################################################################
### file import and mask definition
#################################################################
#---------------------
### import Pupil, Mask (FPM), and Lyot stop files
param PupilFile {x in Xs,y in Ys};
read {x in Xs,y in Ys} PupilFile[x,y] < (sprintf("%s%s", directoryAperture, filenameAperture));
close (sprintf("%s%s", directoryAperture, filenameAperture));

param MaskQuarterFile {mx in MXsQ, my in MYsQ};
read {mx in MXsQ, my in MYsQ} MaskQuarterFile[mx,my] < (sprintf("%s%s", directoryFPM, filenameFPM));
close (sprintf("%s%s", directoryFPM, filenameFPM));

param LyotFile0 {x in Xs,y in Ys};
read {x in Xs,y in Ys} LyotFile0[x,y] < (sprintf("%s%s", directoryLyot,filenameLyotStop0));
close (sprintf("%s%s", directoryLyot, filenameLyotStop0));

param LyotFile1 {x in Xs,y in Ys};
read {x in Xs,y in Ys} LyotFile1[x,y] < (sprintf("%s%s", directoryLyot,filenameLyotStop1));
close (sprintf("%s%s", directoryLyot, filenameLyotStop1));

param LyotFile2 {x in Xs,y in Ys};
read {x in Xs,y in Ys} LyotFile2[x,y] < (sprintf("%s%s", directoryLyot,filenameLyotStop2));
close (sprintf("%s%s", directoryLyot, filenameLyotStop2));

#---------------------
### definition of set of points where Pupil, Mask, and Lyot are non null
set Pupil := setof {x in Xs, y in Ys: PupilFile[x,y] != 0.} (x,y);
#set Mask := setof {mx in MXs, my in MYs: MaskFile[mx,my] != 0.} (mx,my);
set Lyot0 := setof {x in Xs, y in Ys: LyotFile0[x,y] != 0} (x,y);
set Lyot1 := setof {x in Xs, y in Ys: LyotFile1[x,y] != 0} (x,y);
set Lyot2 := setof {x in Xs, y in Ys: LyotFile2[x,y] != 0} (x,y);

set PupilQuarter := setof {x in XsQ, y in YsQ: PupilFile[x,y] != 0} (x,y);
set MaskQuarter := setof {mx in MXsQ, my in MYsQ: MaskQuarterFile[mx,my] != 0} (mx,my);

#---------------------
### transmission of the Pupil. Used for calibration.
param TR := sum {x in XsQ, y in YsQ} PupilFile[x,y]*dx*dy;

#################################################################
### propagation through APLC coronagraph
#################################################################
#---------------------
### variable A apodizer with transmission to optimize, values ranging between 0 and 1, default value 0.5
var A_quarter {x in XsQ, y in YsQ} >= 0, <= 1, := 0.5; #fix this
var A {x in Xs, y in Ys} >= 0, <= 1, := 0.5; #fix this
subject to st_A_1 {x in XsQ, y in YsQ}: A[x,y]  =A_quarter[x,y];
subject to st_A_2 {x in XsQ, y in YsQ}: A[-x,y] =A_quarter[x,y];
subject to st_A_3 {x in XsQ, y in YsQ}: A[x,-y] =A_quarter[x,y];
subject to st_A_4 {x in XsQ, y in YsQ}: A[-x,-y]=A_quarter[x,y];

#---------------------
### electric field within the FPM in plane B
var EBm_cx_quarter {x in XsQ, my in MYsQ, lam in Ls} := 0.0;
var EBm_real_quarter {mx in MXsQ, my in MYsQ, lam in Ls} := 0.0;

subject to st_EBm_cx_quarter {x in XsQ, my in MYsQ, lam in Ls}: EBm_cx_quarter[x,my,lam] = 2.*sum {y in YsQ: (x,y) in PupilQuarter} A_quarter[x,y]*PupilFile[x,y]*cos(2*pi*y*my*(lam0/lam))*dy;
subject to st_EBm_real_quarter {(mx,my) in MaskQuarter, lam in Ls}: EBm_real_quarter[mx,my,lam] = 2.*(lam0/lam)*sum {x in XsQ} (EBm_cx_quarter[x,my,lam]*cos(2*pi*x*mx*(lam0/lam)))*dx;

#---------------------
### electric field in plane C (before Lyot stop), be careful, FPM is assumed to be binary. If not, add MaskQuarterFile in the expression
var ECm1_Bmreal_cx_quarter {mx in MXsQ, y in YsQ, lam in Ls} := 0.0;
var ECm_real_quarter {x in XsQ, y in YsQ, lam in Ls} := 0.0;

subject to st_ECm1_Bmreal_cx_quarter {mx in MXsQ, y in YsQ, lam in Ls}: ECm1_Bmreal_cx_quarter[mx,y,lam] = 2.*sum {my in MYsQ: (mx,my) in MaskQuarter} EBm_real_quarter[mx,my,lam]*MaskQuarterFile[mx,my]*cos(2*pi*y*my*(lam0/lam))*dmy;
subject to st_ECm_real_quarter {x in XsQ, y in YsQ, lam in Ls}: ECm_real_quarter[x,y,lam] = 2.*(lam0/lam)*sum {mx in MXsQ} (ECm1_Bmreal_cx_quarter[mx,y,lam]*cos(2*pi*x*mx*(lam0/lam)))*dmx;

#---------------------
var ECm_real {x in Xs, y in Ys, lam in Ls} := 0.0;
subject to st_ECm_real_1 {x in XsQ, y in YsQ, lam in Ls}: ECm_real[x,y,lam]  =ECm_real_quarter[x,y,lam];
subject to st_ECm_real_2 {x in XsQ, y in YsQ, lam in Ls}: ECm_real[-x,y,lam] =ECm_real_quarter[x,y,lam];
subject to st_ECm_real_3 {x in XsQ, y in YsQ, lam in Ls}: ECm_real[x,-y,lam] =ECm_real_quarter[x,y,lam];
subject to st_ECm_real_4 {x in XsQ, y in YsQ, lam in Ls}: ECm_real[-x,-y,lam]=ECm_real_quarter[x,y,lam];

#---------------------
### electric field in plane D (final image) for three different Lyot stops
var ED1_ECmreal_cx0 {x in Xs, eta in Etas, lam in Ls} := 0.0;
var ED1_ECmreal_sx0 {x in Xs, eta in Etas, lam in Ls} := 0.0;
var ED1_ECmreal_cx1 {x in Xs, eta in Etas, lam in Ls} := 0.0;
var ED1_ECmreal_sx1 {x in Xs, eta in Etas, lam in Ls} := 0.0;
var ED1_ECmreal_cx2 {x in Xs, eta in Etas, lam in Ls} := 0.0;
var ED1_ECmreal_sx2 {x in Xs, eta in Etas, lam in Ls} := 0.0;
var ED_real0 {xi in Xis, eta in Etas, lam in Ls} := 0.0;
var ED_imag0 {xi in Xis, eta in Etas, lam in Ls} := 0.0;
var ED_real1 {xi in Xis, eta in Etas, lam in Ls} := 0.0;
var ED_imag1 {xi in Xis, eta in Etas, lam in Ls} := 0.0;
var ED_real2 {xi in Xis, eta in Etas, lam in Ls} := 0.0;
var ED_imag2 {xi in Xis, eta in Etas, lam in Ls} := 0.0;

subject to st_ED1_ECmreal_cx0 {x in Xs, eta in Etas, lam in Ls}: ED1_ECmreal_cx0[x,eta,lam] = sum {y in Ys: (x,y) in Lyot0} (A[x,y]*PupilFile[x,y]-ECm_real[x,y,lam])*LyotFile0[x,y]*cos(2*pi*y*eta*(lam0/lam))*dy;
subject to st_ED1_ECmreal_sx0 {x in Xs, eta in Etas, lam in Ls}: ED1_ECmreal_sx0[x,eta,lam] = sum {y in Ys: (x,y) in Lyot0} (A[x,y]*PupilFile[x,y]-ECm_real[x,y,lam])*LyotFile0[x,y]*sin(2*pi*y*eta*(lam0/lam))*dy;
subject to st_ED1_ECmreal_cx1 {x in Xs, eta in Etas, lam in Ls}: ED1_ECmreal_cx1[x,eta,lam] = sum {y in Ys: (x,y) in Lyot1} (A[x,y]*PupilFile[x,y]-ECm_real[x,y,lam])*LyotFile1[x,y]*cos(2*pi*y*eta*(lam0/lam))*dy;
subject to st_ED1_ECmreal_sx1 {x in Xs, eta in Etas, lam in Ls}: ED1_ECmreal_sx1[x,eta,lam] = sum {y in Ys: (x,y) in Lyot1} (A[x,y]*PupilFile[x,y]-ECm_real[x,y,lam])*LyotFile1[x,y]*sin(2*pi*y*eta*(lam0/lam))*dy;
subject to st_ED1_ECmreal_cx2 {x in Xs, eta in Etas, lam in Ls}: ED1_ECmreal_cx2[x,eta,lam] = sum {y in Ys: (x,y) in Lyot2} (A[x,y]*PupilFile[x,y]-ECm_real[x,y,lam])*LyotFile2[x,y]*cos(2*pi*y*eta*(lam0/lam))*dy;
subject to st_ED1_ECmreal_sx2 {x in Xs, eta in Etas, lam in Ls}: ED1_ECmreal_sx2[x,eta,lam] = sum {y in Ys: (x,y) in Lyot2} (A[x,y]*PupilFile[x,y]-ECm_real[x,y,lam])*LyotFile2[x,y]*sin(2*pi*y*eta*(lam0/lam))*dy;


subject to st_ED_real0 {(xi, eta) in CoroOuterRegionAll, lam in Ls}: ED_real0[xi,eta,lam] = (lam0/lam)*sum {x in Xs} (ED1_ECmreal_cx0[x,eta,lam]*cos(2*pi*x*xi*(lam0/lam))-ED1_ECmreal_sx0[x,eta,lam]*sin(2*pi*x*xi*(lam0/lam)))*dx;
subject to st_ED_imag0 {(xi, eta) in CoroOuterRegionAll, lam in Ls}: ED_imag0[xi,eta,lam] = (lam0/lam)*sum {x in Xs} -(ED1_ECmreal_cx0[x,eta,lam]*sin(2*pi*x*xi*(lam0/lam))+ED1_ECmreal_sx0[x,eta,lam]*cos(2*pi*x*xi*(lam0/lam)))*dx;
subject to st_ED_real1 {(xi, eta) in CoroOuterRegionAll, lam in Ls}: ED_real1[xi,eta,lam] = (lam0/lam)*sum {x in Xs} (ED1_ECmreal_cx1[x,eta,lam]*cos(2*pi*x*xi*(lam0/lam))-ED1_ECmreal_sx1[x,eta,lam]*sin(2*pi*x*xi*(lam0/lam)))*dx;
subject to st_ED_imag1 {(xi, eta) in CoroOuterRegionAll, lam in Ls}: ED_imag1[xi,eta,lam] = (lam0/lam)*sum {x in Xs} -(ED1_ECmreal_cx1[x,eta,lam]*sin(2*pi*x*xi*(lam0/lam))+ED1_ECmreal_sx1[x,eta,lam]*cos(2*pi*x*xi*(lam0/lam)))*dx;
subject to st_ED_real2 {(xi, eta) in CoroOuterRegionAll, lam in Ls}: ED_real2[xi,eta,lam] = (lam0/lam)*sum {x in Xs} (ED1_ECmreal_cx2[x,eta,lam]*cos(2*pi*x*xi*(lam0/lam))-ED1_ECmreal_sx2[x,eta,lam]*sin(2*pi*x*xi*(lam0/lam)))*dx;
subject to st_ED_imag2 {(xi, eta) in CoroOuterRegionAll, lam in Ls}: ED_imag2[xi,eta,lam] = (lam0/lam)*sum {x in Xs} -(ED1_ECmreal_cx2[x,eta,lam]*sin(2*pi*x*xi*(lam0/lam))+ED1_ECmreal_sx2[x,eta,lam]*cos(2*pi*x*xi*(lam0/lam)))*dx;

#################################################################
### propagation through the system without coronagraph mask
#################################################################
#---------------------
### electric field peak of the image at lam0 (no coronagraph mask)
var ED00_real0 := 0.0;
subject to st_ED00_real0: ED00_real0 = sum {x in Xs, y in Ys: (x,y) in Lyot0} (A[x,y]*PupilFile[x,y]*LyotFile0[x,y])*dx*dy;

#---------------------
### electric field in plane D (final image) for only the centered Lyot stop (no coronagraph mask)
var ED0_ECmreal_cx0 {x in Xs, eta in Etas, lam in Ls} := 0.0;
var ED0_ECmreal_sx0 {x in Xs, eta in Etas, lam in Ls} := 0.0;
var ED0_ECmreal_cx1 {x in Xs, eta in Etas, lam in Ls} := 0.0;
var ED0_ECmreal_sx1 {x in Xs, eta in Etas, lam in Ls} := 0.0;
var ED0_ECmreal_cx2 {x in Xs, eta in Etas, lam in Ls} := 0.0;
var ED0_ECmreal_sx2 {x in Xs, eta in Etas, lam in Ls} := 0.0;
var ED0_real0 {xi in Xis, eta in Etas, lam in Ls} := 0.0;
var ED0_imag0 {xi in Xis, eta in Etas, lam in Ls} := 0.0;
var ED0_real1 {xi in Xis, eta in Etas, lam in Ls} := 0.0;
var ED0_imag1 {xi in Xis, eta in Etas, lam in Ls} := 0.0;
var ED0_real2 {xi in Xis, eta in Etas, lam in Ls} := 0.0;
var ED0_imag2 {xi in Xis, eta in Etas, lam in Ls} := 0.0;

subject to st_ED0_ECmreal_cx0 {x in Xs, eta in Etas, lam in Ls}: ED0_ECmreal_cx0[x,eta,lam] = sum {y in Ys: (x,y) in Lyot0} A[x,y]*PupilFile[x,y]*LyotFile0[x,y]*cos(2*pi*y*eta*(lam0/lam))*dy;
subject to st_ED0_ECmreal_sx0 {x in Xs, eta in Etas, lam in Ls}: ED0_ECmreal_sx0[x,eta,lam] = sum {y in Ys: (x,y) in Lyot0} A[x,y]*PupilFile[x,y]*LyotFile0[x,y]*sin(2*pi*y*eta*(lam0/lam))*dy;
subject to st_ED0_ECmreal_cx1 {x in Xs, eta in Etas, lam in Ls}: ED0_ECmreal_cx1[x,eta,lam] = sum {y in Ys: (x,y) in Lyot1} A[x,y]*PupilFile[x,y]*LyotFile1[x,y]*cos(2*pi*y*eta*(lam0/lam))*dy;
subject to st_ED0_ECmreal_sx1 {x in Xs, eta in Etas, lam in Ls}: ED0_ECmreal_sx1[x,eta,lam] = sum {y in Ys: (x,y) in Lyot1} A[x,y]*PupilFile[x,y]*LyotFile1[x,y]*sin(2*pi*y*eta*(lam0/lam))*dy;
subject to st_ED0_ECmreal_cx2 {x in Xs, eta in Etas, lam in Ls}: ED0_ECmreal_cx2[x,eta,lam] = sum {y in Ys: (x,y) in Lyot2} A[x,y]*PupilFile[x,y]*LyotFile2[x,y]*cos(2*pi*y*eta*(lam0/lam))*dy;
subject to st_ED0_ECmreal_sx2 {x in Xs, eta in Etas, lam in Ls}: ED0_ECmreal_sx2[x,eta,lam] = sum {y in Ys: (x,y) in Lyot2} A[x,y]*PupilFile[x,y]*LyotFile2[x,y]*sin(2*pi*y*eta*(lam0/lam))*dy;


#################################################################
### optimization problem
#################################################################
#---------------------
### maximization of the apodizer throughput/transmission
maximize throughput: sum{(x,y) in Pupil} A[x,y]*dx*dy/TR;

#---------------------
### constraints in the coronagraphic image core region
### sqrt(2.) factor in the constraints is for margin purpose at all the wavelengths;
# calculations relevant to Lyot Stop 0
subject to sidelobe_zero_real_pos_core0 {(xi,eta) in CoroCoreRegion, lam in Ls}: ED_real0[xi,eta,lam] <= 10^(-cCoreRegion/2)*ED00_real0/sqrt(2.);
subject to sidelobe_zero_real_neg_core0 {(xi,eta) in CoroCoreRegion, lam in Ls}: ED_real0[xi,eta,lam] >= -10^(-cCoreRegion/2)*ED00_real0/sqrt(2.);
subject to sidelobe_zero_imag_pos_core0 {(xi,eta) in CoroCoreRegion, lam in Ls}: ED_imag0[xi,eta,lam] <= 10^(-cCoreRegion/2)*ED00_real0/sqrt(2.);
subject to sidelobe_zero_imag_neg_core0 {(xi,eta) in CoroCoreRegion, lam in Ls}: ED_imag0[xi,eta,lam] >= -10^(-cCoreRegion/2)*ED00_real0/sqrt(2.);

## calculations relevant to Lyot Stop 1
subject to sidelobe_zero_real_pos_core1 {(xi,eta) in CoroCoreRegion, lam in Ls}: ED_real1[xi,eta,lam] <= 10^(-cCoreRegion/2)*ED00_real0/sqrt(2.);
subject to sidelobe_zero_real_neg_core1 {(xi,eta) in CoroCoreRegion, lam in Ls}: ED_real1[xi,eta,lam] >= -10^(-cCoreRegion/2)*ED00_real0/sqrt(2.);
subject to sidelobe_zero_imag_pos_core1 {(xi,eta) in CoroCoreRegion, lam in Ls}: ED_imag1[xi,eta,lam] <= 10^(-cCoreRegion/2)*ED00_real0/sqrt(2.);
subject to sidelobe_zero_imag_neg_core1 {(xi,eta) in CoroCoreRegion, lam in Ls}: ED_imag1[xi,eta,lam] >= -10^(-cCoreRegion/2)*ED00_real0/sqrt(2.);

# calculations relevant to Lyot Stop 2
subject to sidelobe_zero_real_pos_core2 {(xi,eta) in CoroCoreRegion, lam in Ls}: ED_real2[xi,eta,lam] <= 10^(-cCoreRegion/2)*ED00_real0/sqrt(2.);
subject to sidelobe_zero_real_neg_core2 {(xi,eta) in CoroCoreRegion, lam in Ls}: ED_real2[xi,eta,lam] >= -10^(-cCoreRegion/2)*ED00_real0/sqrt(2.);
subject to sidelobe_zero_imag_pos_core2 {(xi,eta) in CoroCoreRegion, lam in Ls}: ED_imag2[xi,eta,lam] <= 10^(-cCoreRegion/2)*ED00_real0/sqrt(2.);
subject to sidelobe_zero_imag_neg_core2 {(xi,eta) in CoroCoreRegion, lam in Ls}: ED_imag2[xi,eta,lam] >= -10^(-cCoreRegion/2)*ED00_real0/sqrt(2.);

#---------------------
### constraints in the dark hole region

# calculations relevant to Lyot Stop 0
subject to sidelobe_zero_real_pos0 {(xi,eta) in CoroDarkHoleRegion, lam in Ls}: ED_real0[xi,eta,lam] <= 10^(-cDarkHoleRegion/2)*ED00_real0/sqrt(2.);
subject to sidelobe_zero_real_neg0 {(xi,eta) in CoroDarkHoleRegion, lam in Ls}: ED_real0[xi,eta,lam] >= -10^(-cDarkHoleRegion/2)*ED00_real0/sqrt(2.);
subject to sidelobe_zero_imag_pos0 {(xi,eta) in CoroDarkHoleRegion, lam in Ls}: ED_imag0[xi,eta,lam] <= 10^(-cDarkHoleRegion/2)*ED00_real0/sqrt(2.);
subject to sidelobe_zero_imag_neg0 {(xi,eta) in CoroDarkHoleRegion, lam in Ls}: ED_imag0[xi,eta,lam] >= -10^(-cDarkHoleRegion/2)*ED00_real0/sqrt(2.);

# calculations relevant to Lyot Stop 1
subject to sidelobe_zero_real_pos1 {(xi,eta) in CoroDarkHoleRegion, lam in Ls}: ED_real1[xi,eta,lam] <= 10^(-cDarkHoleRegion/2)*ED00_real0/sqrt(2.);
subject to sidelobe_zero_real_neg1 {(xi,eta) in CoroDarkHoleRegion, lam in Ls}: ED_real1[xi,eta,lam] >= -10^(-cDarkHoleRegion/2)*ED00_real0/sqrt(2.);
subject to sidelobe_zero_imag_pos1 {(xi,eta) in CoroDarkHoleRegion, lam in Ls}: ED_imag1[xi,eta,lam] <= 10^(-cDarkHoleRegion/2)*ED00_real0/sqrt(2.);
subject to sidelobe_zero_imag_neg1 {(xi,eta) in CoroDarkHoleRegion, lam in Ls}: ED_imag1[xi,eta,lam] >= -10^(-cDarkHoleRegion/2)*ED00_real0/sqrt(2.);

# calculations relevant to Lyot Stop 2
subject to sidelobe_zero_real_pos2 {(xi,eta) in CoroDarkHoleRegion, lam in Ls}: ED_real2[xi,eta,lam] <= 10^(-cDarkHoleRegion/2)*ED00_real0/sqrt(2.);
subject to sidelobe_zero_real_neg2 {(xi,eta) in CoroDarkHoleRegion, lam in Ls}: ED_real2[xi,eta,lam] >= -10^(-cDarkHoleRegion/2)*ED00_real0/sqrt(2.);
subject to sidelobe_zero_imag_pos2 {(xi,eta) in CoroDarkHoleRegion, lam in Ls}: ED_imag2[xi,eta,lam] <= 10^(-cDarkHoleRegion/2)*ED00_real0/sqrt(2.);
subject to sidelobe_zero_imag_neg2 {(xi,eta) in CoroDarkHoleRegion, lam in Ls}: ED_imag2[xi,eta,lam] >= -10^(-cDarkHoleRegion/2)*ED00_real0/sqrt(2.);

#################################################################
### solver for optimization problem
#################################################################
#---------------------
### options for AMPL
option times 1;
option gentimes 1;
option show_stats 1;
option TMPDIR '/ssdfast1/tmp';

#---------------------
### options for gurobi solver with linear programming method
option solver gurobi;
option gurobi_options "outlev=1 lpmethod=2 crossover=0";

#---------------------
#solve and display of the result
solve;
display solve_result_num, solve_result;

display directoryOutput;
display filenameApodizerDat;

#################################################################
### export solution
#################################################################
#---------------------
#export solution A (apodizer/shaped pupil for APLC)
printf {x in Xs, y in Ys}: "%15g %15g %15g \n", x, y, A[x,y] > (sprintf("%s%s", directoryOutput, filenameApodizerDat));
