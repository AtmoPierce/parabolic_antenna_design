use std::f64::consts::PI;
use num_complex::Complex64;

pub struct AntennaDesign {
    // Inputs
    pub focal_length: f64,
    pub wavelength: f64,
    pub d1: f64,
    pub d2: f64,

    // Derived
    pub f_over_d: f64,
    pub psi_0: f64,

    // Outputs (linear)
    pub spl: f64,
    pub atl: f64,
    pub pel: f64,
    pub xol: f64,
    pub directivity: f64,

    // Outputs (decibels)
    pub spl_db: f64,
    pub atl_db: f64,
    pub pel_db: f64,
    pub xol_db: f64,
    pub directivity_db: f64,

    // Settings
    pub steps: usize,
}

impl AntennaDesign {
    pub fn new(focal_length: f64, wavelength: f64, d1: f64, d2: f64, steps: usize) -> Self {
        let f_over_d = focal_length / d2;
        let psi_0 = 2.0 * (1.0 / (4.0 * f_over_d)).atan();
        Self {
            focal_length,
            wavelength,
            d1,
            d2,
            f_over_d,
            psi_0,
            spl: 0.0,
            atl: 0.0,
            pel: 0.0,
            xol: 0.0,
            directivity: 0.0,

            spl_db: 0.0,
            atl_db: 0.0,
            pel_db: 0.0,
            xol_db: 0.0,
            directivity_db: 0.0,

            steps,
        }
    }

    pub fn added_edge_taper_db(&self) -> f64 {
        let cos_squared = (self.psi_0 / 2.0).cos().powi(2);
        20.0 * cos_squared.log10()
    }

    fn rk4_integrate_line<F>(&self, a: f64, b: f64, f: F) -> f64
    where
        F: Fn(f64) -> f64,
    {
        let h = (b - a) / self.steps as f64;
        let mut sum = 0.0;
        let mut x = a;

        for _ in 0..self.steps {
            let k1 = f(x);
            let k2 = f(x + h / 2.0);
            let k3 = f(x + h / 2.0);
            let k4 = f(x + h);

            sum += (k1 + 2.0 * k2 + 2.0 * k3 + k4) * h / 6.0;
            x += h;
        }

        sum
    }

    fn integrate_2d_rk4<F>(&self, psi_min: f64, psi_max: f64, f: F) -> f64
    where
        F: Fn(f64, f64) -> f64,
    {
        let dpsi = (psi_max - psi_min) / self.steps as f64;
        let mut sum = 0.0;

        for i in 0..self.steps {
            let psi0 = psi_min + i as f64 * dpsi;

            let inner_integral = |psi: f64| {
                self.rk4_integrate_line(0.0, 2.0 * PI, |phi| f(psi, phi))
            };

            let k1 = inner_integral(psi0);
            let k2 = inner_integral(psi0 + dpsi / 2.0);
            let k3 = inner_integral(psi0 + dpsi / 2.0);
            let k4 = inner_integral(psi0 + dpsi);

            sum += (k1 + 2.0 * k2 + 2.0 * k3 + k4) * dpsi / 6.0;
        }

        sum
    }

    pub fn amplitude_taper_efficiency<F>(&self, e_field: F) -> f64
    where
        F: Fn(f64, f64) -> f64,
    {
        let numerator = self
            .integrate_2d_rk4(0.0, self.psi_0, |psi, phi| {
                e_field(psi, phi) * (psi / 2.0).tan()
            })
            .powi(2);

        let denom = self.integrate_2d_rk4(0.0, self.psi_0, |psi, phi| {
            e_field(psi, phi).powi(2) * psi.sin()
        });

        let aperture_factor = PI * (self.psi_0 / 2.0).tan().powi(2);
        numerator / (aperture_factor * denom)
    }

    pub fn phase_efficiency<F>(&self, e_field: F) -> f64
    where
        F: Fn(f64, f64) -> f64,
    {
        let num = self
            .integrate_2d_rk4(0.0, self.psi_0, |psi, phi| {
                e_field(psi, phi) * (psi / 2.0).tan()
            })
            .powi(2);

        let den = self.integrate_2d_rk4(0.0, self.psi_0, |psi, phi| {
            e_field(psi, phi).powi(2) * (psi / 2.0).tan()
        });

        num / den
    }

    pub fn spillover_efficiency<F>(&self, e_field: F) -> f64
    where
        F: Fn(f64, f64) -> f64,
    {
        let num = self.integrate_2d_rk4(0.0, self.psi_0, |psi, phi| {
            e_field(psi, phi).powi(2) * psi.sin()
        });

        let den = self.integrate_2d_rk4(0.0, PI, |psi, phi| {
            e_field(psi, phi).powi(2) * psi.sin()
        });

        num / den
    }

    pub fn cross_polarization_efficiency<Fc, Fx>(&self, ec: Fc, ex: Fx) -> f64
    where
        Fc: Fn(f64, f64) -> f64,
        Fx: Fn(f64, f64) -> f64,
    {
        let num = self.integrate_2d_rk4(0.0, PI, |psi, phi| {
            ec(psi, phi).powi(2) * psi.sin()
        });

        let den = self.integrate_2d_rk4(0.0, PI, |psi, phi| {
            (ec(psi, phi).powi(2) + ex(psi, phi).powi(2)) * psi.sin()
        });

        num / den
    }
    pub fn compute<F, Fx>(&mut self, ec: F, ex: Fx)
    where
        F: Fn(f64, f64) -> f64 + Copy,
        Fx: Fn(f64, f64) -> f64 + Copy,
    {
        self.atl = self.amplitude_taper_efficiency(ec);
        self.pel = self.phase_efficiency(ec);
        self.spl = self.spillover_efficiency(ec);
        self.xol = self.cross_polarization_efficiency(ec, ex);

        self.directivity = self.directivity_ratio();

        // dB equivalents
        self.atl_db = to_db(self.atl);
        self.pel_db = to_db(self.pel);
        self.spl_db = to_db(self.spl);
        self.xol_db = to_db(self.xol);

        let geometric_db = to_db((PI / self.wavelength).powi(2) * (self.d2.powi(2) - self.d1.powi(2)));
        self.directivity_db =
            geometric_db + self.spl_db + self.atl_db + self.pel_db + self.xol_db;
    }

    pub fn directivity_ratio(&self) -> f64 {
        let area = (PI / self.wavelength).powi(2) * (self.d2.powi(2) - self.d1.powi(2));
        area * self.spl * self.atl * self.pel * self.xol
    }
}


// Generator functions for E-field
pub fn ec_generator(psi_0: f64, edge_taper_db: f64) -> impl Fn(f64, f64) -> f64 + Copy {
    let taper_linear = 10f64.powf(edge_taper_db / 20.0);
    let cos_psi0 = psi_0.cos();
    let n = taper_linear.ln() / cos_psi0.ln();

    move |psi: f64, _phi: f64| {
        if psi > psi_0 {
            0.0
        } else {
            psi.cos().powf(n)
        }
    }
}

pub fn ec_generator_complex(
    psi_0: f64,
    edge_taper_db: f64,
    focal_length: f64,
    wavelength: f64,
) -> impl Fn(f64, f64) -> Complex64 + Copy {
    let taper_linear = 10f64.powf(edge_taper_db / 20.0);
    let cos_psi0 = psi_0.cos();
    let n = (taper_linear.ln()) / (cos_psi0.ln());

    move |psi: f64, _phi: f64| {
        if psi > psi_0 {
            Complex64::new(0.0, 0.0)
        } else {
            let amplitude = psi.cos().powf(n);
            let delta_L = focal_length * (1.0 / psi.cos() - 1.0);
            let phase = -2.0 * std::f64::consts::PI * delta_L / wavelength;
            Complex64::from_polar(amplitude, phase)
        }
    }
}

pub fn to_db(x: f64) -> f64 {
    if x <= 0.0 {
        f64::NEG_INFINITY // or -300.0 if you want to avoid -inf
    } else {
        10.0 * x.log10()
    }
}