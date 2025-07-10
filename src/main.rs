use std::f64::consts::PI;
use std::fs::File;
use std::io::Write;
use parabolic_antenna_design::{AntennaDesign, ec_generator_complex, ec_generator};

#[derive(Debug, Clone, Copy)]
struct AntennaParams {
    taper_db: f64,
    focal_length: f64,
    d1: f64,
    d2: f64,
}

fn evaluate(params: AntennaParams, wavelength: f64, steps: usize) -> AntennaDesign {
    let psi_0 = (params.d2 / (2.0 * params.focal_length)).atan();
    let psi_penalty = if psi_0 > 45_f64.to_radians() {
        (45_f64.to_radians() / psi_0).powi(2)
    } else {
        1.0
    };

    let ec_complex = ec_generator_complex(psi_0, params.taper_db, params.focal_length, wavelength);
    let ec = move |psi: f64, phi: f64| ec_complex(psi, phi).re;

    let ec = ec_generator(psi_0, params.taper_db);
    let mut ex = |_, _| 0.0;

    
    let mut antenna = AntennaDesign::new(params.focal_length, wavelength, params.d1, params.d2, steps);
    antenna.compute(ec, ex);
    antenna.psi_0 = psi_0;
    antenna.f_over_d = params.focal_length / params.d2;
    antenna
}

fn penalty(x: f64, threshold: f64) -> f64 {
    if x < threshold {
        (x / threshold).powi(2) // penalize smoothly below threshold
    } else {
        1.0
    }
}
fn penalty_fd(f_over_d: f64, min_fd: f64) -> f64 {
    if f_over_d < min_fd {
        // Quadratic penalty as f/D drops below min_fd
        (f_over_d / min_fd).powi(2)
    } else {
        1.0
    }
}

fn gradient_descent(
    mut params: AntennaParams,
    wavelength: f64,
    steps: usize,
    learning_rate: f64,
    iterations: usize,
) -> AntennaDesign {
    let mut best_score = f64::NEG_INFINITY;
    let mut best_params = params;

    let f = |p: AntennaParams| {
        let antenna = evaluate(p, wavelength, steps);
        let atl_penalty = penalty(antenna.atl, 0.3);
        let pel_penalty = penalty(antenna.pel, 0.3);
        antenna.directivity_db
    };

    for _ in 0..iterations {
        let h = 1e-4;

        let grad = (
            (f(AntennaParams { taper_db: params.taper_db + h, ..params }) - f(AntennaParams { taper_db: params.taper_db - h, ..params })) / (2.0 * h),
            (f(AntennaParams { focal_length: params.focal_length + h, ..params }) - f(AntennaParams { focal_length: params.focal_length - h, ..params })) / (2.0 * h),
            (f(AntennaParams { d1: params.d1 + h, ..params }) - f(AntennaParams { d1: params.d1 - h, ..params })) / (2.0 * h),
            (f(AntennaParams { d2: params.d2 + h, ..params }) - f(AntennaParams { d2: params.d2 - h, ..params })) / (2.0 * h),
        );

        params.taper_db     -= learning_rate * grad.0;
        params.focal_length -= learning_rate * grad.1;
        params.d1           -= learning_rate * grad.2;
        params.d2           -= learning_rate * grad.3;

        params.taper_db = params.taper_db.clamp(-18.0, -8.0);
        let d2_min = wavelength * 5.5;
        let d2_max = 1.0;
        params.d2 = params.d2.clamp(d2_min, d2_max);

        let fd_min = 0.5;
        let fd_max = 0.8;
        let fd = (params.focal_length / params.d2).clamp(fd_min, fd_max);
        params.focal_length = fd * params.d2;

        let d1_min = wavelength * 1.0;
        let d1_max = wavelength * 2.0;
        let d1_upper = (params.d2 * 0.25).min(d1_max);
        params.d1 = params.d1.clamp(d1_min, d1_upper);


        let score_now = f(params);
        if score_now > best_score {
            best_score = score_now;
            best_params = params;
        }
    }

    evaluate(best_params, wavelength, steps)
}

fn export_geometry_csv(focal_length: f64, d2: f64, d1: f64, filename: &str) -> std::io::Result<()> {
    let mut file = File::create(filename)?;
    writeln!(file, "r,z")?;

    let n = 100;
    let r_max = d2 / 2.0;

    for i in 0..=n {
        let r = i as f64 * r_max / n as f64;
        let z = r * r / (4.0 * focal_length);
        writeln!(file, "{:.6},{:.6}", r, z)?;
    }

    // Feed position (at focal point)
    writeln!(file, "# feed center")?;
    writeln!(file, "{:.6},{:.6}", 0.0, focal_length)?;
    writeln!(file, "# feed radius")?;
    writeln!(file, "{:.6},{:.6}", d1 / 2.0, focal_length)?;

    Ok(())
}

fn export_svg(focal_length: f64, d2: f64, d1: f64, filename: &str) -> std::io::Result<()> {
let mut file = File::create(filename)?;
    let n = 200;

    let r_max = d2 / 2.0;
    let z_max = r_max * r_max / (4.0 * focal_length);
    let feed_z = focal_length;
    let feed_length = 0.04; // 4cm
    let horn_radius = d1 / 2.0;

    let z_total = z_max + feed_z + feed_length;
    let r_total = r_max.max(horn_radius);

    writeln!(
        file,
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="{min_x} {min_y} {w} {h}" width="500" height="500">"#,
        min_x = -r_total,
        min_y = -(z_total * 1.1), // extra space at top
        w = r_total * 2.0,
        h = z_total * 1.2
    )?;

    // Dish shape
    writeln!(file, r#"<polyline fill="none" stroke="black" stroke-width="0.002" points=""#)?;
    for i in 0..=n {
        let r = i as f64 * r_max / n as f64;
        let z = r * r / (4.0 * focal_length);
        writeln!(file, "{:.4},{:.4} ", r, -z)?;
    }
    writeln!(file, r#"" />"#)?;

    // Feed horn shape as a triangle
    let horn_base_y = -feed_z;
    let horn_tip_y = -feed_z - feed_length;
    writeln!(
        file,
        r#"<polygon points="{:.4},{:.4} {:.4},{:.4} {:.4},{:.4}" fill="red" stroke="black" stroke-width="0.001"/>"#,
        -horn_radius, horn_base_y,
        horn_radius, horn_base_y,
        0.0, horn_tip_y,
    )?;

    writeln!(file, "</svg>")?;
    Ok(())
}

fn export_openscad(focal_length: f64, d2: f64, d1: f64, filename: &str) -> std::io::Result<()> {
    let mut file = File::create(filename)?;
    let r_max = d2 / 2.0;
    let n = 100;

    writeln!(file, "points = [")?;
    for i in 0..=n {
        let r = i as f64 * r_max / n as f64;
        let z = r * r / (4.0 * focal_length);
        writeln!(file, "[{:.6}, {:.6}],", r, z)?;
    }
    writeln!(file, "];")?;

    writeln!(file, r#"
        rotate_extrude($fn=200)
            polygon(points);
        "#)?;

    Ok(())
}

fn export_openscad_with_feed(
    focal_length: f64,
    d2: f64,
    d1: f64,
    filename: &str,
) -> std::io::Result<()> {
    let mut file = File::create(filename)?;
    let r_max = d2 / 2.0;
    let n = 100;

    // Reflector profile
    writeln!(file, "// Paraboloid profile points")?;
    writeln!(file, "points = [")?;
    for i in 0..=n {
        let r = i as f64 * r_max / n as f64;
        let z = r * r / (4.0 * focal_length);
        writeln!(file, "  [{:.6}, {:.6}],", r, z)?;
    }
    writeln!(file, "];\n")?;

    // Top-level render first
    writeln!(file, r#"
        union() {{
            reflector();
            feed_horn();
        }}
        "#)?;

            // Reflector module
            writeln!(file, r#"
        module reflector() {{
            rotate_extrude($fn=200)
                polygon(points);
        }}"#)?;

            // Feed horn
            let horn_length = 0.04;
            let horn_tip_radius = 0.001;
            let horn_radius = d1 / 2.0;

            writeln!(file, r#"
        module feed_horn() {{
            translate([0, 0, {:.6}])
                rotate_extrude($fn=100)
                    polygon([
                        [{:.6}, 0],
                        [{:.6}, {:.6}]
                    ]);
        }}"#, focal_length, horn_tip_radius, horn_radius, -horn_length)?;

    Ok(())
}




fn main() {
    let wavelength = 0.0125;
    let steps = 100;

    let initial_params = AntennaParams {
        taper_db: -20.0,
        focal_length: 0.15,
        d1: 0.05,
        d2: 0.30,
    };

    let antenna = gradient_descent(
        initial_params,
        wavelength,
        steps,
        0.0001,
        200,
    );

    println!("\nOptimal Design Parameters:");
    println!("ψ₀ (deg): {:.2}", antenna.psi_0.to_degrees());
    println!("Edge Taper (dB): {:.2}", antenna.added_edge_taper_db());
    println!("f/D: {:.2}", antenna.f_over_d);
    println!("Focal Length (m): {:.4}", antenna.focal_length);
    println!("Feed Diameter d1 (m): {:.4}", antenna.d1);
    println!("Reflector Diameter d2 (m): {:.4}", antenna.d2);

    println!("\nPerformance Metrics:");
    println!("Directivity: {:.2} dB", antenna.directivity_db);
    println!("SPL: {:.4} ({:.2} dB)", antenna.spl, antenna.spl_db);
    println!("ATL: {:.4} ({:.2} dB)", antenna.atl, antenna.atl_db);
    println!("PEL: {:.4} ({:.2} dB)", antenna.pel, antenna.pel_db);
    println!("XOL: {:.4} ({:.2} dB)", antenna.xol, antenna.xol_db);

    export_geometry_csv(
        antenna.focal_length,
        antenna.d2,
        antenna.d1,
        "antenna_geometry.csv",
    ).expect("Failed to export geometry");

    export_svg(antenna.focal_length, antenna.d2, antenna.d1, "antenna_profile.svg")
        .expect("Failed to write SVG");

    export_openscad_with_feed(antenna.focal_length, antenna.d2, antenna.d1, "antenna_feed.scad")
        .expect("Failed to write OpenSCAD file");
}
