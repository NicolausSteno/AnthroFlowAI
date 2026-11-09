# d16_common.R -- D16 shared setup: Chapter 3 fitted model (slade_cycle_drop_core, objective 807.96)
# Simulating-Resources (thesis Chapter 3) must be cloned next to AnthroFlowAI, or its path given in
# the environment variable SIMULATING_RESOURCES_DIR. Run the D16 scripts from final_model/.
SR <- Sys.getenv("SIMULATING_RESOURCES_DIR", normalizePath(file.path(getwd(), "..", "..", "Simulating-Resources"), mustWork = FALSE))
OUT <- file.path("discussion", "outputs"); dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
.libPaths(c(list.files(file.path(SR, "renv/library"), full.names = TRUE)[1] |> list.files(full.names = TRUE), .libPaths()))
suppressMessages(library(deSolve))
source(file.path(SR, "model_v2.R")); source(file.path(SR, "gloser_data_v2.R"))
d <- comparison_data
# fitted vector, from params_full__slade_cycle_drop_core.json
FIT <- list(c = 8.93968, phi = 0.682005, C0 = 0.347912, c1 = -0.0656737, c2 = 2e-4,
            eps_Y = 1.1, eps_P = 3.99353e-06, k_E = 0.353272, z_E = 0.85, k_R = 0.55,
            z_R = 0.0314783, n = 2.19189e-06, s5_ref = 1, alpha9 = 0.0269607,
            gamma = 0.555125, alpha3 = 20.0644, alpha8 = 105.051, lam4 = 3e-4, theta = 0.3)
mkp <- function(...) {
  p <- reference_parms(); for (nm in names(FIT)) p[[nm]] <- FIT[[nm]]
  ov <- list(...); for (nm in names(ov)) p[[nm]] <- ov[[nm]]
  p$alpha[9] <- p$alpha9; p$alpha[3] <- p$alpha3; p$alpha[8] <- p$alpha8; p$lambda[4] <- p$lam4
  p
}
# data_y0() of run_all.R (adjust_s0 = true), power limiter
mky0 <- function(p, Rmax0 = NULL, E0 = NULL) {
  r <- reference_y0(); i1 <- which.min(d$Year)
  g <- function(col) { x <- d[[col]][i1]; if (is.finite(x)) x else NA_real_ }
  keep <- function(new, old, lo = 1e-3, hi = 1e4) if (is.finite(new) && new > lo && new < hi) new else old
  s <- r$s
  s[1] <- keep(g("Extraction (Tg/a)") / (p$alpha[1] + p$alpha[4]), s[1])
  s[2] <- keep(g("F2 (Tg/a)") / p$alpha[2], s[2])
  s[3] <- keep(g("Semis (Tg/a)") / (p$alpha[3] + p$alpha[8]), s[3])
  s[5] <- keep(p$s5_ref * (g("Recycling (Tg/a)") / r$Rmax)^(1 / p$n), s[5], lo = 1e-3, hi = 1e3)
  r$s <- s
  if (!is.null(Rmax0)) r$Rmax <- Rmax0
  if (!is.null(E0)) r$E <- E0
  r
}
solve1 <- function(p, y0 = mky0(p), T = 61, tol = 1e-8) {
  o <- NULL
  invisible(capture.output(suppressWarnings(
    o <- try(dmmcm_v2(p, y0, seq(0, T, 1), funGDP, "slade", eps_sw = .02,
                      atol = tol, rtol = tol, maxsteps = 20000), silent = TRUE))))
  if (inherits(o, "try-error") || is.null(o) || nrow(o) < T + 1 || any(!is.finite(as.matrix(o)))) NULL else o
}
