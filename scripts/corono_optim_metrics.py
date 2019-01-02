import numpy as np
import pylab as pl
import os


def get_metrics(self, fp2res=16, rho_out=None, Nlam=None, use_gray_gap_zero=True, verbose=True): # for APLC class
        TelAp_basename = 
        gapstr_beg = TelAp_basename.find('gap')
        TelAp_nopad_basename = TelAp_basename.replace(TelAp_basename[gapstr_beg:gapstr_beg+4], 'gap0')
        TelAp_nopad_fname = os.path.join( os.path.dirname(self.fileorg['TelAp fname']), TelAp_nopad_basename )
        #if self.design['Pupil']['edge'] == 'floor': # floor to binary
        #    TelAp_p = np.floor(np.loadtxt(self.fileorg['TelAp fname'])).astype(int)
        #elif self.design['Pupil']['edge'] == 'round': # round to binary
        #    TelAp_p = np.round(np.loadtxt(self.fileorg['TelAp fname'])).astype(int)
        #else:
        #    TelAp_p = np.loadtxt(self.fileorg['TelAp fname'])
        if os.path.exists(TelAp_nopad_fname) and use_gray_gap_zero:
            TelAp_p = np.loadtxt(TelAp_nopad_fname)
            telap_flag = 0
        else:
            TelAp_p = np.loadtxt(self.fileorg['TelAp fname'])
            telap_flag = 1
        A_col = np.loadtxt(self.fileorg['sol fname'])[:,-1]
        LS_p = np.loadtxt(self.fileorg['LS fname'])
        A_p = A_col.reshape(TelAp_p.shape)
        if isinstance(self, QuarterplaneAPLC):
            TelAp = np.concatenate((np.concatenate((TelAp_p[::-1,::-1], TelAp_p[:,::-1]),axis=0),
                                    np.concatenate((TelAp_p[::-1,:], TelAp_p),axis=0)), axis=1)
            A = np.concatenate((np.concatenate((A_p[::-1,::-1], A_p[:,::-1]),axis=0),
                                np.concatenate((A_p[::-1,:], A_p),axis=0)), axis=1)
            LS = np.concatenate((np.concatenate((LS_p[::-1,::-1], LS_p[:,::-1]),axis=0),
                                 np.concatenate((LS_p[::-1,:], LS_p),axis=0)), axis=1)
        elif isinstance(self, HalfplaneAPLC):
            TelAp = np.concatenate((TelAp_p[:,::-1], TelAp_p), axis=1)
            A = np.concatenate((A_p[:,::-1], A_p), axis=1)
            LS = np.concatenate((LS_p[:,::-1], LS_p), axis=1)
        else:
            TelAp = TelAp_p
            A = A_p
            LS = LS_p

        self.eval_metrics['apod nb res ratio'] = np.sum(np.abs(A - np.round(A)))/np.sum(TelAp)
        # Account for the ratio of the diameter of the square enclosing the aperture to the circumscribed circle
        if self.design['Pupil']['prim'] in self._square2circ_ratio:
            D = self._square2circ_ratio[self.design['Pupil']['prim']]
        else:
            D = 1.
        N = self.design['Pupil']['N']
        if Nlam is None:
            Nlam = self.design['Image']['Nlam']
        if rho_out is None:
            rho_out = self.design['Image']['oda']
        dx = (D/2)/N
        dy = dx
        xs = np.matrix(np.linspace(-N+0.5, N-0.5, 2*N)*dx)
        ys = xs.copy()
        M_fp2 = int(np.ceil(rho_out*fp2res))
        dxi = 1./fp2res
        xis = np.matrix(np.linspace(-M_fp2+0.5, M_fp2-0.5, 2*M_fp2)*dxi)
        etas = xis.copy()
        wrs = np.linspace(1.-self.design['Image']['bw']/2, 1.+self.design['Image']['bw']/2, Nlam)
        XXs = np.asarray(np.dot(np.matrix(np.ones(xis.shape)).T, xis))
        YYs = np.asarray(np.dot(etas.T, np.matrix(np.ones(etas.shape))))
        RRs = np.sqrt(XXs**2 + YYs**2)
        p7ap_ind = np.less_equal(RRs, 0.7)

        intens_D_0_polychrom = np.zeros((Nlam, 2*M_fp2, 2*M_fp2))
        intens_D_0_peak_polychrom = np.zeros((Nlam, 1))
        intens_TelAp_polychrom = np.zeros((Nlam, 2*M_fp2, 2*M_fp2))
        intens_TelAp_peak_polychrom = np.zeros((Nlam, 1))
        for wi, wr in enumerate(wrs):
            Psi_D_0 = dx*dy/wr*np.dot(np.dot(np.exp(-1j*2*np.pi/wr*np.dot(xis.T, xs)), TelAp*A*LS[::-1,::-1]),
                                             np.exp(-1j*2*np.pi/wr*np.dot(xs.T, xis)))
            intens_D_0_polychrom[wi] = np.power(np.absolute(Psi_D_0), 2)
            intens_D_0_peak_polychrom[wi] = (np.sum(TelAp*A*LS[::-1,::-1])*dx*dy/wr)**2
            Psi_TelAp = dx*dy/wr*np.dot(np.dot(np.exp(-1j*2*np.pi/wr*np.dot(xis.T, xs)), TelAp),
                                               np.exp(-1j*2*np.pi/wr*np.dot(xs.T, xis)))
            intens_TelAp_polychrom[wi] = np.power(np.absolute(Psi_TelAp), 2)
            intens_TelAp_peak_polychrom[wi] = (np.sum(TelAp)*dx*dy/wr)**2

        intens_D_0 = np.mean(intens_D_0_polychrom, axis=0)
        intens_D_0_peak = np.mean(intens_D_0_peak_polychrom)
        intens_TelAp = np.mean(intens_TelAp_polychrom, axis=0)
        intens_TelAp_peak = np.mean(intens_TelAp_peak_polychrom)

        fwhm_ind_APLC = np.greater_equal(intens_D_0, intens_D_0_peak/2)
        fwhm_ind_TelAp = np.greater_equal(intens_TelAp, intens_TelAp_peak/2)

        fwhm_sum_TelAp = np.sum(intens_TelAp[fwhm_ind_TelAp])*dxi*dxi
        fwhm_sum_APLC = np.sum(intens_D_0[fwhm_ind_APLC])*dxi*dxi
        p7ap_sum_TelAp = np.sum(intens_TelAp[p7ap_ind])*dxi*dxi
        p7ap_sum_APLC = np.sum(intens_D_0[p7ap_ind])*dxi*dxi

        self.eval_metrics['inc energy'] = np.sum(np.power(TelAp,2)*dx*dx)
        self.eval_metrics['tot thrupt'] = np.sum(intens_D_0*dxi*dxi)/np.sum(np.power(TelAp,2)*dx*dx)
        self.eval_metrics['fwhm thrupt'] = fwhm_sum_APLC/np.sum(np.power(TelAp,2)*dx*dx)
        self.eval_metrics['fwhm circ thrupt'] = fwhm_sum_APLC/(np.pi/4)
        self.eval_metrics['p7ap thrupt'] = p7ap_sum_APLC/np.sum(np.power(TelAp,2)*dx*dx)
        self.eval_metrics['p7ap circ thrupt'] = p7ap_sum_APLC/(np.pi/4)
        self.eval_metrics['rel fwhm thrupt'] = fwhm_sum_APLC/fwhm_sum_TelAp
        self.eval_metrics['rel p7ap thrupt'] = p7ap_sum_APLC/p7ap_sum_TelAp
        self.eval_metrics['fwhm area'] = np.sum(fwhm_ind_APLC)*dxi*dxi
        if verbose:
            print("////////////////////////////////////////////////////////")
            print("{:s}".format(self.fileorg['job name']))
            print("Incident energy on aperture (dimensionless): {:.3f}".format(self.eval_metrics['inc energy']))
            print("Non-binary residuals, as a percentage of clear telescope aperture area: {:.2f}%".format(100*self.eval_metrics['apod nb res ratio']))
            print("Band-averaged total throughput: {:.2f}%".format(100*self.eval_metrics['tot thrupt']))
            print("Band-averaged half-max throughput: {:.2f}%".format(100*self.eval_metrics['fwhm thrupt']))
            print("Band-averaged half-max throughput, circ. ref.: {:.2f}%".format(100*self.eval_metrics['fwhm circ thrupt']))
            print("Band-averaged r=.7 lam/D throughput: {:.2f}%".format(100*self.eval_metrics['p7ap thrupt']))
            print("Band-averaged r=.7 lam/D throughput, circ. ref.: {:.2f}%".format(100*self.eval_metrics['p7ap circ thrupt']))
            print("Band-averaged relative half-max throughput: {:.2f}%".format(100*self.eval_metrics['rel fwhm thrupt']))
            print("Band-averaged relative r=0.7 lam/D throughput: {:.2f}%".format(100*self.eval_metrics['rel p7ap thrupt']))
            print("Band-averaged FWHM PSF area / (lambda0/D)^2: {:.2f}".format(self.eval_metrics['fwhm area']))
        return telap_flag
