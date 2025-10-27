//! Utility functions for parameter transformations.

/// Epsilon for clamping values during inverse transforms (logit/log)
pub const PARAM_EPSILON: f64 = 1e-8;

/// Standard sigmoid function: maps (-inf, inf) -> (0, 1)
pub fn sigmoid(phi: f64) -> f64 {
    1.0 / (1.0 + (-phi).exp())
}

/// Logit function (inverse sigmoid): maps (0, 1) -> (-inf, inf)
pub fn logit(theta: f64) -> f64 {
    // Clamp to avoid log(0) or division by zero
    let theta_clamped = theta.clamp(PARAM_EPSILON, 1.0 - PARAM_EPSILON);
    (theta_clamped / (1.0 - theta_clamped)).ln()
}

/// Scaled sigmoid: maps (-inf, inf) -> (a, b)
pub fn scaled_sigmoid(phi: f64, a: f64, b: f64) -> f64 {
    a + (b - a) * sigmoid(phi)
}

/// Inverse scaled sigmoid: maps (a, b) -> (-inf, inf)
pub fn inv_scaled_sigmoid(theta: f64, a: f64, b: f64) -> f64 {
    let y = (theta - a) / (b - a);
    logit(y)
}

/// Exponential transform: maps (-inf, inf) -> (0, inf)
pub fn exp_transform(phi: f64) -> f64 {
    phi.exp()
}

/// Log transform (inverse exp): maps (0, inf) -> (-inf, inf)
pub fn inv_exp_transform(theta: f64) -> f64 {
    theta.clamp(PARAM_EPSILON, f64::MAX).ln()
}
