\documentclass[12pt]{article}

\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage{bm}
\usepackage{graphicx}
\usepackage{url}
\usepackage{geometry}
\geometry{a4paper,margin=2.5cm}

\begin{document}

%--------------------------------------------------------------------
% Title and front matter
%--------------------------------------------------------------------
\title{Calculation of differential cross sections for electron impact excitation of H and He$^+$}

\author{Zhuo-Jin Lai, De-Feng Chen, Lin-Qing Pan, Xiao-Han Jiang,\\
Yong-Liang Xu, and Zhang-Jin Chen\thanks{Corresponding author. Email: chenzj@stu.edu.cn}}

\date{J. At. Mol. Sci. 5 (2014) 311--323\\[4pt]
Received 28 February 2014; Accepted (in revised version) 14 May 2014;\\
Published Online 29 October 2014}

\maketitle

\begin{abstract}
We present the distorted wave Born approximation (DWBA) for electron impact excitation and a method to calibrate the DWBA. With the calibrated DWBA, the differential cross sections (DCS) for excitation of H and He$^+$ from $1s$ to $2s$ and $2p$ are calculated and the results are compared with the absolute experimental measurements for H at incident energies of 50 eV and 100 eV. It has been found that the theoretical results are in very good agreement with the experiment, which confirms the validity of the calibration procedure. This work prepares an efficient theoretical method for numerical simulations of non-sequential double ionization of He in strong laser pulse in which laser-induced electron impact excitation of He$^+$ is involved.
\end{abstract}

\noindent\textbf{PACS}: 34.80.Dp, 34.50.Rp

\noindent\textbf{Key words}: electron impact excitation, distorted wave Born approximation, differential cross sections, total cross sections

%--------------------------------------------------------------------
\section{Introduction}
%--------------------------------------------------------------------

The process of electron impact excitation of atoms and ions is one of the most basic and important processes in atomic physics. Theoretical investigations of such problems are of not only practical interest but also more fundamental interest. Numerous theoretical methods have been proposed for calculations of differential cross sections (DCS) for electron impact excitation, including the distorted wave Born approximation (DWBA) \,[1,2], the second-order distorted wave model \,[3], the convergent close-coupling (CCC) calculations \,[5], and the R-matrix method \,[4], among which the DWBA is the simplest. The sophisticated theoretical models, such as the CCC and the R-matrix method, are supposed to be able to reproduce accurate DCS in angular distribution and absolute magnitude as well at low incident energies. On the other hand, for high energies, both the total cross sections (TCS) and the DCS predicted by DWBA are in fairly good agreement with the absolute measurements. However, it has been well recognized that, at low energies, the TCS predicted by the DWBA substantially overestimates the experimental values. ``Ideally, one could use the R-matrix approach for low energies, the DWBA for high energies, and the two theories would yield the same results for intermediate energies. Unfortunately, we do not live in an ideal world'' \,[6].

The purpose of this work is to calibrate the DWBA for electron impact excitation of H and He$^+$ at low energies by employing the empirical formula proposed by Tong et al.\,[7]. This calibration procedure has been previously applied to correct the overestimate of DWBA on the DCS for electron impact excitation of Ne and Ar \,[8].

Our ultimate objective is to apply the calibrated DWBA to simulate the correlated momentum distributions in nonsequential double ionization (NSDI) of He in strong laser fields \,[9,10]. The process of NSDI is one of the laser-induced rescattering processes, which still remains one of the most interesting and challenging topics in strong field physics. Both electron impact ionization and electron impact excitation of ions could be involved in NSDI. In the last two decades a lot of experimental measurements have been performed, particularly noteworthy are the correlated momentum distributions of the two outgoing electrons which were measured at the turn of this century \,[11]. In the meantime, a number of theoretical efforts have been devoted to this problem as well. In one of the theoretical models, which was developed by Chen et al.\,[12,13], the correlated two-electron momentum spectra can be treated as a product of the wave packet for laser-induced returning electrons and the differential cross sections for the laser-free electron impact excitation and/or ionization of the parent ion. In the practical simulations of the correlated electron momentum distributions for NSDI, one needs to evaluate the DCS for electron impact excitation of the parent ion to all possible excited states at all incident energies from threshold to the maximum returning electron energy which is usually less than 200 eV. Due to the heavy computational demand, relatively simple and efficient theoretical approaches are highly desired. Since the shape of the DCS predicted by DWBA is typically in fairly good agreement with the experimental measurements, once the overestimate of DWBA on the DCS is corrected, the calibrated DWBA can serve as a good candidate for such required theoretical tools.

The organization of this paper is as follows: In Section~2, the theory of DWBA for electron impact excitation is presented in detail and the method to calibrate DWBA is proposed. In Section~3, the normalization factors for DWBA at incident energies below 1000 eV are given for electron impact excitation of H and He$^+$ from $1s$ to $2s$ and $2p$, and the calibrated DCS of DWBA for H at 50 eV and 100 eV are compared with the experimental measurements. Furthermore, some calibrated DCS of DWBA for H and He$^+$ at four different incident energies below 100 eV are analyzed. Finally, some conclusions are drawn in Section~4.

Atomic units are used in this paper unless otherwise specified.

%--------------------------------------------------------------------
\section{Theory}
%--------------------------------------------------------------------

In this section, we present the general form for DWBA theory in detail on electron impact excitation of atoms, which can be easily applied to electron impact excitation of ions. A method used to calibrate the DWBA at energies below 1000 eV is also given.

%--------------------------------------------------------------------
\subsection{Basic equations}
%--------------------------------------------------------------------

The problem to be considered here is inelastic electron--atom (e--A) scattering. The Hamiltonian for such a process is given by
\begin{equation}
H = -\frac{1}{2}\nabla^2_{\mathbf{r}_1} + V_{A^+}(\mathbf{r}_1)
    - \frac{1}{2}\nabla^2_{\mathbf{r}_2} + V_{A^+}(\mathbf{r}_2)
    + \frac{1}{r_{12}},
\label{eq:H}
\end{equation}
where $\mathbf{r}_1$ and $\mathbf{r}_2$ are the position vectors for the projectile and the bound-state electron with respect to the nucleus, respectively. In Eq.~\eqref{eq:H}, $V_{A^+}$ is the effective potential based on the single-active-electron approximation, which takes the form
\begin{equation}
V_{A^+}(r)
  = -\frac{1 + a_1 e^{-a_2 r} + a_3 r e^{-a_4 r} + a_5 e^{-a_6 r}}{r},
\label{eq:VA}
\end{equation}
where the parameters $a_i$ (given explicitly in Table~1 in Tong and Lin \,[14]) were obtained by fitting the calculated binding energies of the ground state and the first few excited states of the target atom using this potential to the experimental data. Both the exact initial-state wave function $\Psi_i(\mathbf{r}_1,\mathbf{r}_2)$ and the final-state wave function $\Psi_f(\mathbf{r}_1,\mathbf{r}_2)$ of the system satisfy the Schrödinger equation
\begin{equation}
H \Psi_j(\mathbf{r}_1,\mathbf{r}_2) = E\,\Psi_j(\mathbf{r}_1,\mathbf{r}_2),
\qquad j = i,f,
\label{eq:Schro}
\end{equation}
where $E$ is the total energy.

Since Eq.~\eqref{eq:Schro} cannot be solved analytically, one has to employ approximate Hamiltonians, which can be expressed as
\begin{equation}
H_j = -\frac{1}{2}\nabla_1^2 + U_j(\mathbf{r}_1)
      -\frac{1}{2}\nabla_2^2 + V_{A^+}(\mathbf{r}_2),
\qquad j = i,f,
\label{eq:Hj}
\end{equation}
where $U_i$ ($U_f$) is the distorting potential used to calculate the wave function $\chi_{k_i}$ ($\chi_{k_f}$) for the projectile in the incident (exit) channel with momentum $k_i$ ($k_f$). With this approximation, the initial (final) state wave function can be expressed as a product of the initial (final) state wave function for the projectile and the wave function for the bound electron in the ground (excited) state.

The initial and final state wave functions for the projectile satisfy the differential equation
\begin{equation}
\left(
 -\frac{1}{2}\nabla_1^2 + U_j(\mathbf{r}_1)
\right)\chi_{k_j}(\mathbf{r}_1)
 = \frac{k_j^2}{2}\,\chi_{k_j}(\mathbf{r}_1),
\qquad j = i,f,
\label{eq:cont}
\end{equation}
and the bound-state wave functions are eigenfunctions of
\begin{equation}
\left(
 -\frac{1}{2}\nabla_2^2 + V_{A^+}(\mathbf{r}_2)
\right)\Phi_j(\mathbf{r}_2)
= \epsilon_j \Phi_j(\mathbf{r}_2),
\qquad j = i,f,
\label{eq:bound}
\end{equation}
where $\epsilon_j$ ($j=i,f$) are the corresponding eigenenergies of the initial and final states. Due to energy conservation,
\begin{equation}
\frac{k_i^2}{2} + \epsilon_i = \frac{k_f^2}{2} + \epsilon_f.
\label{eq:energy-conservation}
\end{equation}

In the distorted wave Born approximation, the \emph{direct} transition amplitude for excitation from an initial state $\Phi_i$ to a final state $\Phi_f$ is given by
\begin{equation}
f = \left\langle
  \chi_{k_f}^{(-)}(\mathbf{r}_1)\,\Phi_f(\mathbf{r}_2)
  \left| V_i \right|
  \Phi_i(\mathbf{r}_2)\,\chi_{k_i}^{(+)}(\mathbf{r}_1)
  \right\rangle,
\label{eq:f-def}
\end{equation}
where $V_i$ is the perturbation interaction,
\begin{equation}
V_i = H - H_i = \frac{1}{r_{12}} + V_{A^+}(\mathbf{r}_1) - U_i(\mathbf{r}_1).
\label{eq:Vi}
\end{equation}
The \emph{exchange} scattering amplitude is given by
\begin{equation}
g = \left\langle
  \Phi_f(\mathbf{r}_1)\,\chi_{k_f}^{(-)}(\mathbf{r}_2)
  \left| V_i \right|
  \Phi_i(\mathbf{r}_2)\,\chi_{k_i}^{(+)}(\mathbf{r}_1)
  \right\rangle.
\label{eq:g-def}
\end{equation}

%--------------------------------------------------------------------
\subsection{Partial wave expansions}
%--------------------------------------------------------------------

To evaluate the scattering amplitudes, we perform standard partial-wave expansions. The distorted wave for the incident electron with outgoing ($+$) boundary condition is expanded as
\begin{equation}
\chi_{k_i}^{(+)}(\mathbf{r}_1)
 = \sqrt{\frac{2}{\pi}}
   \frac{1}{k_i r_1}
   \sum_{l_i \mu_i} i^{\,l_i}
   \chi_{l_i}(k_i,r_1)\,
   Y_{l_i\mu_i}(\hat{\mathbf{r}}_1)\,
   Y_{l_i\mu_i}^*(\hat{\mathbf{k}}_i),
\label{eq:chi-plus}
\end{equation}
where $\hat{\mathbf{r}}_1$ and $\hat{\mathbf{k}}_i$ are the unit vectors in the directions of $\mathbf{r}_1$ and $\mathbf{k}_i$, and $Y_{lm}$ are the spherical harmonics. Similarly, the partial-wave expansion for the scattering electron with incoming ($-$) boundary condition is
\begin{equation}
\left[\chi_{k_f}^{(-)}(\mathbf{r}_1)\right]^*
 = \sqrt{\frac{2}{\pi}}
   \frac{1}{k_f r_1}
   \sum_{l_f \mu_f} i^{-l_f}
   \chi_{l_f}(k_f,r_1)\,
   Y_{l_f\mu_f}(\hat{\mathbf{r}}_1)\,
   Y_{l_f\mu_f}^*(\hat{\mathbf{k}}_f).
\label{eq:chi-minus}
\end{equation}
In this work, all continuum waves are normalized to $\delta(k-k')$. For a plane wave, the radial component $\chi_l(k,r)/(k r)$ in Eqs.~\eqref{eq:chi-plus} and \eqref{eq:chi-minus} reduces to a standard spherical Bessel function $j_l(kr)$.

The initial and final bound states can be expressed as
\begin{equation}
\Phi_j(\mathbf{r}_2)
  = \frac{1}{r_2}\,\phi_{N_j L_j}(r_2)\,Y_{L_j M_j}(\hat{\mathbf{r}}_2),
\qquad j = i,f.
\label{eq:Phi}
\end{equation}

In the scattering amplitude \eqref{eq:f-def}, the perturbation potential is the last remaining quantity which needs to be expanded. The first term in the perturbation potential \eqref{eq:Vi} can be expanded as
\begin{equation}
\frac{1}{r_{12}}
 = 4\pi \sum_{l_T \mu_T}
   e_{l_T}^{-2}\,
   \frac{r_<^{\,l_T}}{r_>^{\,l_T+1}}\,
   Y_{l_T\mu_T}^*(\hat{\mathbf{r}}_1)\,
   Y_{l_T\mu_T}(\hat{\mathbf{r}}_2),
\label{eq:1overr12}
\end{equation}
where $e_l = \sqrt{2l+1}$, and $r_< = \min(r_1,r_2)$, $r_> = \max(r_1,r_2)$. Actually, the second and third terms of the perturbation potential in Eq.~\eqref{eq:Vi} are spherically symmetric and can be expressed as
\begin{equation}
V_{A^+}(r_1)-U_i(r_1)
  = 4\pi \sum_{l_T\mu_T}
    \bigl[ V_{A^+}(r_1)-U_i(r_1) \bigr]\,
    Y_{l_T\mu_T}^*(\hat{\mathbf{r}}_1)\,
    Y_{l_T\mu_T}(\hat{\mathbf{r}}_2)\,
    \delta_{l_T 0}.
\label{eq:VA-Ui}
\end{equation}
The expansions \eqref{eq:1overr12} and \eqref{eq:VA-Ui} then yield
\begin{equation}
V_i(\mathbf{r}_1,\mathbf{r}_2)
 = 4\pi\sum_{l_T\mu_T}
   e_{l_T}^{-2}\,A_{l_T}(r_1,r_2)\,
   Y_{l_T\mu_T}^*(\hat{\mathbf{r}}_1)\,
   Y_{l_T\mu_T}(\hat{\mathbf{r}}_2),
\label{eq:Vi-exp}
\end{equation}
where the radial factor $A_{l_T}(r_1,r_2)$ is given by
\begin{equation}
A_{l_T}(r_1,r_2)
 = \frac{r_<^{\,l_T}}{r_>^{\,l_T+1}}
   + \bigl[ V_{A^+}(r_1)-U_i(r_1) \bigr]\delta_{l_T 0}.
\label{eq:AlT}
\end{equation}

%--------------------------------------------------------------------
\subsection{Calculation of the differential cross sections}
%--------------------------------------------------------------------

The differential cross section for electron impact excitation of atoms is given by
\begin{equation}
\frac{d\sigma}{d\Omega}
 = N (2\pi)^4 \frac{k_f}{k_i}\,
   \frac{1}{2L_i+1}
   \times \sum_{M_i=-L_i}^{L_i}
   \sum_{M_f=-L_f}^{L_f}
   \left[
     \frac{3}{4}\,|f-g|^2
     +\frac{1}{4}\,|f+g|^2
   \right],
\label{eq:DCS}
\end{equation}
where the prefactor $N$ denotes the number of electrons in the subshell from which one electron is excited.

With the expansions given in the above subsection, the direct scattering amplitude is given by
\begin{equation}
\begin{split}
f =\;&
  8 \sum_{l_f\mu_f} i^{-l_f}
    \sum_{l_T\mu_T} e_{l_T}^{-2}
    \sum_{l_i\mu_i} i^{\,l_i}\,
    Y_{l_f\mu_f}^*(\hat{\mathbf{k}}_f)\,
    Y_{l_i\mu_i}^*(\hat{\mathbf{k}}_i)\,
    \\[2pt]
   &\times
   \frac{1}{k_f k_i}
  \int dr_1\,dr_2\;
  \chi_{l_f}(k_f,r_1)\,
  \phi_{N_f L_f}^*(r_2)\,
  \\[2pt]
   &\times
  A_{l_T}(r_1,r_2)\,
  \phi_{N_i L_i}(r_2)\,
  \chi_{l_i}(k_i,r_1)\,
  \\[2pt]
   &\times
  F_1 F_2,
\end{split}
\label{eq:f-expanded}
\end{equation}
where $F_1$ and $F_2$ are given by
\begin{equation}
F_1 \equiv
\int d\hat{\mathbf{r}}_1\,
  Y_{l_f\mu_f}(\hat{\mathbf{r}}_1)\,
  Y_{l_i\mu_i}(\hat{\mathbf{r}}_1)\,
  Y_{l_T\mu_T}^*(\hat{\mathbf{r}}_1)
 =
 \frac{1}{\sqrt{4\pi}}\,
 \frac{e_{l_f} e_{l_i}}{e_{l_T}} \,
 C(l_f l_i l_T;\mu_f,\mu_i,\mu_T)\,
 (l_f l_i l_T;000),
\label{eq:F1}
\end{equation}
and
\begin{equation}
F_2 \equiv
\int d\hat{\mathbf{r}}_2\,
  Y_{l_T\mu_T}(\hat{\mathbf{r}}_2)\,
  Y_{L_i M_i}(\hat{\mathbf{r}}_2)\,
  Y_{L_f M_f}^*(\hat{\mathbf{r}}_2)
 =
 \frac{1}{\sqrt{4\pi}}\,
 \frac{e_{l_T} e_{L_i}}{e_{L_f}} \,
 C(l_T L_i L_f;\mu_T,M_i,M_f)\,
 (l_T L_i L_f;000).
\label{eq:F2}
\end{equation}
To perform the integrals over polar angles in Eqs.~\eqref{eq:F1} and \eqref{eq:F2}, we have used the relations
\begin{equation}
Y_{l,-m}(\hat{\mathbf{r}})
 = (-)^m Y_{lm}^*(\hat{\mathbf{r}})
\label{eq:Y-conj}
\end{equation}
and
\begin{equation}
\int
  Y_{l_1 m_1}(\hat{\mathbf{r}})
  Y_{l_2 m_2}(\hat{\mathbf{r}})
  Y_{l_3 m_3}(\hat{\mathbf{r}})
\,d\hat{\mathbf{r}}
 =
 \frac{1}{\sqrt{4\pi}}\,
 \frac{e_{l_1} e_{l_2}}{e_{l_3}} \,
 C(l_1 l_2 l_3;m_1,m_2,-m_3)\,
 (l_1 l_2 l_3;000)\,
 (-)^{m_3},
\label{eq:YYY}
\end{equation}
where $C(l_1 l_2 l_3;m_1 m_2 m_3)$ is a Clebsch--Gordan coefficient.

The product of $F_1$ and $F_2$ can be further simplified as
\begin{equation}
\begin{split}
F_1 F_2 =\;&
\frac{1}{4\pi}\,
\frac{e_{l_f} e_{l_i} e_{L_i}}{e_{L_f}} \,
C(l_f l_i l_T;000)\,
C(l_T L_i L_f;000)
\\
&\times
\sum_{g}
  e_{l_T}\,
  e_g W(l_f l_i L_f L_i; l_T g)\,
  \\
&\times
  C(l_i L_i g; \mu_i, M_i, \mu_g)\,
  C(l_f g L_f; \mu_f,\mu_g,M_f),
\end{split}
\label{eq:F1F2-1}
\end{equation}
where $W(l_1 l_2 l_3 l_4; l_5 l_6)$ is a Racah coefficient, and we have used
\begin{equation}
\begin{split}
C(l_1 l_2 l_3; m_1 m_2 m_3)\,
C(l_3 l_4 l_5; m_3 m_4 m_5)
 =\;&
 \\
\sum_{g}
  e_{l_3}\,
  e_g W(l_1 l_2 l_5 l_4; l_3 g)\,
  C(l_2 l_4 g; m_2, m_4, \mu_g)\,
  C(l_1 g l_5; m_1, \mu_g, m_5).
\end{split}
\label{eq:CG-W}
\end{equation}
Furthermore, by using
\begin{equation}
C(l_1 l_2 l_3; m_1 m_2 m_3)
 = (-)^{l_1-m_1}\,
   \frac{e_{l_3}}{e_{l_2}}\,
   C(l_1 l_3 l_2; m_1, -m_3, -m_2),
\label{eq:CG-rel}
\end{equation}
we rewrite $C(l_f g L_f; \mu_f, \mu_g, M_f)$ in Eq.~\eqref{eq:F1F2-1} as
\begin{equation}
C(l_f g L_f; \mu_f, \mu_g, M_f)
 = (-)^{l_f-\mu_f}\,
   \frac{e_{L_f}}{e_g}\,
   C(l_f L_f g; \mu_f, -M_f, -\mu_g).
\label{eq:CG-rewrite}
\end{equation}
Consequently,
\begin{equation}
\begin{split}
F_1 F_2 =\;&
\frac{1}{4\pi}\,
(-)^{l_f-\mu_f}\,
e_{l_f} e_{l_i} e_{L_i} e_{l_T}\,
C(l_f l_i l_T;000)\,
C(l_T L_i L_f;000)
\\
&\times
\sum_{g}
  W(l_f l_i L_f L_i; l_T g)\,
  C(l_i L_i g; \mu_i, M_i, \mu_i+M_i)\,
\\
&\qquad\qquad\times
  C(l_f L_f g; \mu_f, -M_f, \mu_f-M_f)\,
  \delta_{\mu_i+M_i,\,M_f-\mu_f}.
\end{split}
\label{eq:F1F2-final}
\end{equation}

Substituting Eq.~\eqref{eq:F1F2-final} into Eq.~\eqref{eq:f-expanded}, we finally obtain
\begin{equation}
\begin{split}
f =\;&
 \frac{2}{\pi}
 \sum_{l_i\mu_i}
 \sum_{l_f\mu_f}
 \sum_{l_T}
 \sum_{g}
 i^{\,l_i + l_f}\,
 \frac{e_{l_f} e_{l_i} e_{L_i}}{e_{l_T}} \,
 C(l_i L_i g; \mu_i, M_i, \mu_i+M_i)
\\
&\times
 C(l_f L_f g; \mu_f, -M_f, \mu_f-M_f)\,
 C(l_f l_i l_T;000)\,
 C(l_T L_i L_f;000)\,
\\
&\times
 W(l_f l_i L_f L_i; l_T g)
 Y_{l_f,-\mu_f}(\hat{\mathbf{k}}_f)\,
 Y_{l_i\mu_i}^*(\hat{\mathbf{k}}_i)\,
 \delta_{\mu_i+M_i,\,M_f-\mu_f}
\\
&\times
 \frac{1}{k_f k_i}
 \int dr_1\,dr_2\;
 \chi_{l_f}(k_f,r_1)\,
 \phi_{N_f L_f}^*(r_2)\,
 A_{l_T}(r_1,r_2)\,
 \phi_{N_i L_i}(r_2)\,
 \chi_{l_i}(k_i,r_1).
\end{split}
\label{eq:f-final}
\end{equation}
Similarly, the exchange scattering amplitude is given by
\begin{equation}
\begin{split}
g =\;&
 \frac{2}{\pi}
 \sum_{l_i\mu_i}
 \sum_{l_f\mu_f}
 \sum_{l_T}
 \sum_{g}
 i^{\,l_i-l_f}\,
 (-)^{L_f+M_f}\,
 \frac{e_{L_f} e_{l_i} e_{L_i}}{e_{l_T}} \,
 C(l_i L_i g; \mu_i, M_i, \mu_i+M_i)
\\
&\times
 C(L_f l_f g; -M_f, -\mu_f, -M_f-\mu_f)\,
 C(L_f l_i l_T;000)\,
 C(l_T L_i l_f;000)\,
\\
&\times
 W(L_f l_i l_f L_i; l_T g)
 Y_{l_f\mu_f}^*(\hat{\mathbf{k}}_f)\,
 Y_{l_i\mu_i}^*(\hat{\mathbf{k}}_i)\,
 \delta_{\mu_i+M_i,\,M_f+\mu_f}
\\
&\times
 \frac{1}{k_f k_i}
 \int dr_1\,dr_2\;
 \phi_{N_f L_f}^*(r_1)\,
 \chi_{l_f}(k_f,r_2)\,
 A_{l_T}(r_1,r_2)\,
 \phi_{N_i L_i}(r_2)\,
 \chi_{l_i}(k_i,r_1).
\end{split}
\label{eq:g-final}
\end{equation}

%--------------------------------------------------------------------
\subsection{Distorting potentials}
%--------------------------------------------------------------------

In the DWBA model, the distorting potentials $U_i$ and $U_f$, which are used in Eq.~\eqref{eq:cont} to evaluate the wave functions for the projectile in the initial and final states respectively, play an important role in the numerical calculations, since the calculated DCS are sensitive to the distorted wave functions describing the projectile. Unfortunately, neither $U_i$ nor $U_f$ is determined directly by the formalism. Here, we use static potentials which take the form
\begin{equation}
U_j(r_1)
  = V_{A^+}(r_1) +
    \int d\mathbf{r}_2\,
    \frac{|\Phi_j(\mathbf{r}_2)|^2}{r_{12}},
    \qquad j = i,f.
\label{eq:Uj}
\end{equation}
As shown previously, $V_{A^+}(r)$ in Eq.~\eqref{eq:Uj} is the atomic potential used to evaluate eigenstate wave functions $\Phi_i$ and $\Phi_f$ for the bound electron in the initial and final states, respectively. Obviously, the distorting potentials given by Eq.~\eqref{eq:Uj} for electron impact excitation of atoms are neutral asymptotically.

%--------------------------------------------------------------------
\subsection{Calibration of DWBA}
%--------------------------------------------------------------------

To evaluate the total cross sections for electron impact excitation, Tong et al.\,[7] employed an empirical formula
\begin{equation}
\sigma_{\mathrm{Tong}}(E_i)
  = \frac{\pi}{\Delta E^2}\,
    \exp\!\left[\frac{1.5(\Delta E-\epsilon)}{E_i}\right]\,
    f\!\left(\frac{E_i}{\Delta E}\right),
\label{eq:sigmaTong}
\end{equation}
where
\begin{equation}
f(x)
  = \frac{1}{x}\left[
      \beta \ln x
      + \gamma\left(1-\frac{1}{x}\right)
      + \delta\,\frac{\ln x}{x}
    \right],
\label{eq:f-of-x}
\end{equation}
with $\Delta E$ the excitation energy for a given transition and $\epsilon$ the eigenenergy of the corresponding excited state in atomic hydrogen. The parameters $\beta$, $\gamma$ and $\delta$ in Eq.~\eqref{eq:f-of-x} have been obtained initially by fitting the TCS to the convergent close-coupling (CCC) results for hydrogen from $1s$ to $2p$ and further tested for $e^- + \mathrm{He}^+(1s) \to e^- + \mathrm{He}^+(2p)$.

However, it has been found that, with the parameters given in Ref.~[7], the formula \eqref{eq:sigmaTong} fails to predict the correct values of the TCS for excitation of other atoms and ions. Even for H and He$^+$, the TCS for excitation to other excited states reproduced by Eq.~\eqref{eq:sigmaTong} are much higher than the CCC data and the shape of the TCS as a function of incident energy does not agree with the CCC very well either.

To adjust the overall difference in magnitude, we introduce a prefactor $\alpha$ to modify the empirical formula, which is given by
\begin{equation}
\sigma_{\mathrm{M\mbox{-}Tong}}(E_i)
  = \alpha\,
    \frac{\pi}{\Delta E^2}\,
    \exp\!\left[\frac{1.5(\Delta E-\epsilon)}{E_i}\right]\,
    f\!\left(\frac{E_i}{\Delta E}\right).
\label{eq:sigmaMTong}
\end{equation}
It should be noted that, in Eq.~\eqref{eq:sigmaMTong}, $\epsilon$ denotes the eigenenergy of the corresponding excited state in target atoms or ions. In the present work, we apply the same fitting procedure as in Ref.~[7] to obtain the parameters. For excitations of H and He$^+$ from $1s$ to $2s$, the parameters we obtained are
\[
\beta = 0.7638,\qquad
\gamma = 1.1759,\qquad
\delta = 0.6706,
\]
which are different from those in Ref.~[7]. It has been found that with this set of parameters, the TCS reproduced by Eq.~\eqref{eq:sigmaMTong} are in better agreement with CCC in shape. These parameters are further tested by comparing the predicted excitation cross section with CCC for excitations of H and He$^+$ from $1s$ to $3s$ and $4s$. For excitations of H and He$^+$ from $1s$ to $np$ ($n=2,3,4$), the parameters are
\[
\beta = 1.32,\qquad
\gamma = -1.08,\qquad
\delta = -0.04.
\]
The prefactor $\alpha$ is then determined by matching the TCS from Eq.~\eqref{eq:sigmaMTong} with the CCC data at high energies.

It should also be noted that the TCS of CCC are not available for most atoms or ions. Hence, the applicability of the above fitting procedure to excitation of other atoms and ions is quite limited. Fortunately, both the DCS and the TCS of DWBA are reliable at high energies. Therefore, the prefactor $\alpha$ can be obtained by matching the TCS from Eq.~\eqref{eq:sigmaMTong} with the DWBA results at high energies, say 1000 eV, provided that the parameters $\beta$, $\gamma$ and $\delta$ remain the same for all target atoms and ions.

The total cross section of DWBA at fixed incident energy $E_i = k_i^2/2$ can be obtained by integrating the DCS of Eq.~\eqref{eq:DCS} over scattering angles:
\begin{equation}
\sigma_{\mathrm{DWBA}}(E_i)
 = \int \frac{d\sigma}{d\Omega}\,d\hat{\mathbf{k}}_f.
\label{eq:sigmaDWBA}
\end{equation}
To calibrate the DWBA at low energies, we define a normalization factor
\begin{equation}
C(E_i)
 = \frac{\sigma_{\mathrm{M\mbox{-}Tong}}(E_i)}
        {\sigma_{\mathrm{DWBA}}(E_i)}.
\label{eq:C}
\end{equation}
By multiplying the DCS of DWBA by the normalization factor at each incident energy, one obtains the calibrated DWBA as
\begin{equation}
\left(\frac{d\sigma}{d\Omega}\right)_C
 = C(E_i)\,\frac{d\sigma}{d\Omega}.
\label{eq:DCS-calibrated}
\end{equation}

%--------------------------------------------------------------------
\section{Results and discussion}
%--------------------------------------------------------------------

To obtain the normalization factors to calibrate the DCS of DWBA, we calculate the TCS from the empirical formula of Eq.~\eqref{eq:sigmaMTong} and the TCS of DWBA. The results are shown in Figs.~\ref{fig:H-TCS} and \ref{fig:He-TCS} for excitations of H and He$^+$, respectively. The corresponding CCC results \,[15] are also plotted for comparison. It can be seen that the CCC data for excitation from $1s$ to $2s$ are reproduced very well for both H and He$^+$ while for excitation from $1s$ to $2p$ slight differences exist. The agreement between the TCS of Tong and CCC can be improved if the TCS of Tong at 1000 eV is fitted to CCC rather than DWBA. The reason that we fit the TCS of Tong at 1000 eV to DWBA instead of CCC is that DWBA results are always available.

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.8\textwidth]{fig1}
  \caption{TCSs (left vertical axis) and normalization factors of DWBA (right vertical axis) for excitation of H from (a) $1s$ to $2s$ and (b) $1s$ to $2p$ at incident energies from the excitation energy of 10.2 eV to 1000 eV. Dotted curve: total cross sections of DWBA; solid curve: total cross sections calculated using the empirical formula Eq.~\eqref{eq:sigmaMTong}; chain curve: normalization factor given by Tong/DWBA; solid circles: CCC data \,[15].}
  \label{fig:H-TCS}
\end{figure}

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.8\textwidth]{fig2}
  \caption{TCSs (left vertical axis) and normalization factors of DWBA (right vertical axis) for excitation of He$^+$ from (a) $1s$ to $2s$ and (b) $1s$ to $2p$ at incident energies from the excitation energy of 40.8 eV to 1000 eV. Dotted curve: total cross sections of DWBA; solid curve: total cross sections calculated using the empirical formula Eq.~\eqref{eq:sigmaMTong}; chain curve: normalization factor given by Tong/DWBA; solid circles: CCC data \,[16].}
  \label{fig:He-TCS}
\end{figure}

The absolute experimental measurements of Khakoo et al.\,[16] for electron impact excitation of the $1^2S \to 2^2S+2^2P$ levels of H at incident energies of 50 and 100 eV provide an excellent possibility of a stringent test for the present calibration procedure. It is illustrated in Fig.~\ref{fig:H-exp} that the calibrated DWBA DCS follow the experimental data very well over the whole angular region for both incident energies.

To see the contributions from the excitations of $1^2S\to 2^2S$ and $1^2S\to 2^2P$ separately, the corresponding theoretical DCS of the calibrated DWBA are also plotted in Fig.~\ref{fig:H-exp} for comparison. One can see that the excitation of $1s$ to $2p$ dominates the forward scattering for the angular region from $0^\circ$ to $45^\circ$ at 50 eV and from $0^\circ$ to $30^\circ$ at 100 eV. On the other hand, the excitation of $1s$ to $2s$ cannot be neglected in the region of larger scattering angles.

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.8\textwidth]{fig3}
  \caption{Comparison of the DCS of the calibrated DWBA with experimental measurements of Khakoo et al.\,[16] for excitation of H from the ground state to the $n=2$ state at incident energies of (a) 50 eV and (b) 100 eV.}
  \label{fig:H-exp}
\end{figure}

In Fig.~\ref{fig:H-diff} we show the differential cross sections of DWBA weighted by the normalization factors for excitations of H from $1s$ to $2s$ and from $1s$ to $2p$ at incident energies of 15, 25, 50 and 100 eV, respectively. The slope of the DCS for both excitations $1s \to 2s$ and $1s \to 2p$ changes more rapidly at larger scattering angles as the incident energy decreases. In addition to the slope change, extra minima are reproduced by the DWBA around $70^\circ$ for excitation of $1s$ to $2p$ and around $100^\circ$ for $1s$ to $2p$ at 15 eV.

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.8\textwidth]{fig4}
  \caption{Calibrated DWBA differential cross sections for excitation of H from (a) $1s$ to $2s$ and (b) $1s$ to $2p$ at incident energies of 15, 25, 50 and 100 eV, respectively.}
  \label{fig:H-diff}
\end{figure}

Figure~\ref{fig:He-diff} shows similar results for excitations of He$^+$ at energies below 100 eV. Compared to the excitation of H, enhanced backward-scattering DCS are predicted by the DWBA due to larger Coulomb attraction to the scattered electron, since large-angle scattering takes place when the projectile approaches closer to the nucleus of He such that it sees more charge than the nuclear charge of H. As a result, a minimum appears in the DCS for both excitation of $1s$ to $2s$ and $1s$ to $2p$ at 45 and 60 eV. Both the depths and positions of the minimum in the DCS have significant physical importance since they reflect the structural information of the targets. In addition, with increase of incident energy, the angle at which the slope changes does not move as much as that for H, which even almost remains fixed at $110^\circ$ for the excitation of He$^+$ from $1s$ to $2p$, as shown in Fig.~\ref{fig:He-diff}(b).

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.8\textwidth]{fig5}
  \caption{Calibrated DWBA differential cross sections for excitation of He$^+$ from (a) $1s$ to $2s$ and (b) $1s$ to $2p$ at incident energies of 45, 60, 80 and 100 eV, respectively.}
  \label{fig:He-diff}
\end{figure}

%--------------------------------------------------------------------
\section{Conclusions}
%--------------------------------------------------------------------

We present a method to correct the overestimate of DWBA on the TCS for electron impact excitation of H and He$^+$. The purpose of this work is to apply the calibrated DWBA to simulate the correlated momentum distributions for laser-induced nonsequential double ionization of He. The calibration method is based on two assumptions: (1) the relative angular distributions of the DCS predicted by the DWBA at low incident energies are fairly accurate, and (2) the TCS reproduced by the DWBA at high incident energies are reliable. The validity of the calibration method is confirmed by the agreement between the DCS obtained by the calibrated DWBA and the absolute experimental measurements for excitations of H from the ground state to the $n=2$ state. The calculated DCS with the calibrated DWBA for excitations of H and He$^+$ from $1s$ to $2s$ and from $1s$ to $2p$ below 100 eV are also presented and the structure of the DCS is analyzed.

%--------------------------------------------------------------------
\section*{Acknowledgments}
%--------------------------------------------------------------------

This work was supported by the National Natural Science Foundation of China under Grant No.~11274219, the STU Scientific Research Foundation for Talents, and the Scientific Research Foundation for the Returned Overseas Chinese Scholars, State Education Ministry.

%--------------------------------------------------------------------
\begin{thebibliography}{99}
%--------------------------------------------------------------------

\bibitem{MadisonShelton1973}
D.~H.~Madison and W.~N.~Shelton,
Phys.\ Rev.\ A \textbf{7} (1973) 499.

\bibitem{BartschatMadison1987}
K.~Bartschat and D.~H.~Madison,
J.\ Phys.\ B \textbf{20} (1987) 5839.

\bibitem{MadisonWinters1983}
D.~H.~Madison and K.~H.~Winters,
J.\ Phys.\ B \textbf{16} (1983) 4437.

\bibitem{ZemanBartschat1997}
V.~Zeman and K.~Bartschat,
J.\ Phys.\ B \textbf{30} (1997) 4609.

\bibitem{BrayStelbovics1992}
I.~Bray and A.~T.~Stelbovics,
Phys.\ Rev.\ A \textbf{46} (1992) 6995.

\bibitem{Khakoo2002}
M.~A.~Khakoo, J.~Wrkich, M.~Larsen, G.~Kleiban, I.~Kanik, S.~Trajmar,
M.~J.~Brunger, P.~J.~O.~Teubner, A.~Crowe, C.~J.~Fontes, R.~E.~H.~Clark,
V.~Zeman, K.~Bartschat, D.~H.~Madison, R.~Srivastava, and A.~D.~Stauffer,
Phys.\ Rev.\ A \textbf{65} (2002) 062711.

\bibitem{TongZhaoLin2003}
X.~M.~Tong, Z.~X.~Zhao, and C.~D.~Lin,
Phys.\ Rev.\ A \textbf{68} (2003) 043412.

\bibitem{Liang2011}
Y.~Q.~Liang, Z.~J.~Chen, D.~H.~Madison, and C.~D.~Lin,
J.\ Phys.\ B \textbf{44} (2011) 085201.

\bibitem{Staudte2007}
A.~Staudte, C.~Ruiz, M.~Sch\"offler, S.~Sch\"ossler, D.~Zeidler,
Th.~Weber, M.~Meckel, D.~M.~Villeneuve, P.~B.~Corkum, A.~Becker,
and R.~D\"orner,
Phys.\ Rev.\ Lett.\ \textbf{99} (2007) 263002.

\bibitem{Rudenko2007}
A.~Rudenko, V.~L.~B.~de~Jesus, Th.~Ergler, K.~Zrost, B.~Feuerstein,
C.~D.~Schr\"oter, R.~Moshammer, and J.~Ullrich,
Phys.\ Rev.\ Lett.\ \textbf{99} (2007) 263003.

\bibitem{Weber2000}
T.~Weber, H.~Giessen, M.~Weckenbrock, G.~Urbasch, A.~Staudte,
L.~Spielberger, O.~Jagutzki, V.~Mergel, M.~Vollmer, and R.~D\"orner,
Nature \textbf{405} (2000) 658.

\bibitem{ChenPRL2010}
Z.~J.~Chen, Y.~Q.~Liang, and C.~D.~Lin,
Phys.\ Rev.\ Lett.\ \textbf{104} (2010) 253201.

\bibitem{ChenPRA2010}
Z.~J.~Chen, Y.~Q.~Liang, and C.~D.~Lin,
Phys.\ Rev.\ A \textbf{82} (2010) 063417.

\bibitem{TongLin2005}
X.~M.~Tong and C.~D.~Lin,
J.\ Phys.\ B \textbf{38} (2005) 2593.

\bibitem{CCCdatabase}
S.~I.~Bray,
CCC-database, \url{http://atom.curtin.edu.au/CCC-WWW/index.html}.

\bibitem{Khakoo1999}
M.~A.~Khakoo, M.~Larsen, B.~Paolini, X.~Guo, I.~Bray, A.~Stelbovics,
I.~Kanik, and S.~Trajmar,
Phys.\ Rev.\ Lett.\ \textbf{82} (1999) 3980.

\end{thebibliography}

\end{document}
