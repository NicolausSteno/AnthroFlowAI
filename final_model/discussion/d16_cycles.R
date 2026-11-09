# d16_cycles.R -- D16 (thesis Discussion, sec:thdisc_cycles): does a larger secondary share damp or
# amplify the copper price cycle in the fitted Chapter 3 model (slade_cycle_drop_core, obj. 807.96)?
# Run from final_model/:  Rscript discussion/d16_cycles.R   (then Rscript discussion/d16_summarise.R)
# Nothing is refitted. Model code: Simulating-Resources/model_v2.R, gloser_data_v2.R (sourced, unedited).
#
# LEVER  sigma0 = Rmax0/(E0+Rmax0), the 1961 secondary share, at FIXED 1961 total supply
#        E0 + Rmax0 = 4.899 Tg/a (observed); s1(0) = E0/(alpha1+alpha4) keeps the mine stock consistent.
#        Crossed with tau_R in {1,2,4,8}, tau_E in {4,8,12,16} yr (chapter values 2 and 8).
# HISTORICAL FORCING (World Bank GDP growth), 1961-2022:
#   amp_long   sd of band-passed ln P, run_all.R band() = loess(.25) - loess(.90): ~15-56 yr band
#   amp_short  sd of ln P - loess(.25): the sub-15 yr band the chapter filter removes
#   per_long / per_short  2*pi/omega from the logfreq statistic of each band
#   sd_dlnE, sd_dlnR, sd_dlnS  sd of annual log growth of extraction, recycling, total supply
# CONSTANT FORCING (GDP growth held at its 1961-2022 mean, 3.47%/yr), 1961-2100:
#   pers_long / pers_short  sd of the band over 2041-2100 / sd over 1961-2020 (fixed-year loess spans):
#                           < 1 the endogenous cycle dies out, ~1 it persists
# INFEASIBILITY: with the fitted n ~ 2e-6 recycling is not limited by scrap, so when R exceeds the
#   end-of-life scrap inflow s5 is exhausted and the solve stops; t_fail and R/f9 at failure recorded.
source(file.path("discussion", "d16_common.R")); suppressMessages(library(parallel))
S0   <- d[["Extraction (Tg/a)"]][1] + d[["Recycling (Tg/a)"]][1]
GBAR <- mean(gdp_growth); fconst <- function(t) GBAR
lo_y <- function(x, yrs) { N <- length(x); as.numeric(predict(loess(x ~ seq_len(N), span = min(yrs / N, 1), degree = 1))) }
# 62-yr window: spans .25 and .90 of 62 points = 15.5 and 55.8 yr, identical to run_all.R band()
long_b  <- function(x) lo_y(x, 15.5) - lo_y(x, 55.8)
short_b <- function(x) x - lo_y(x, 15.5)
perf <- function(b) 2 * pi / sqrt(sum(diff(b)^2) / sum(b^2))

y0_share <- function(p, sig) { y <- mky0(p, Rmax0 = sig * S0, E0 = (1 - sig) * S0)
  y$s[1] <- y$E / (p$alpha[1] + p$alpha[4]); if (p$n > 1e-3) y$s[5] <- 1; y }

raw_solve <- function(p, y0, T, f) { o <- NULL
  invisible(capture.output(suppressWarnings(o <- try(dmmcm_v2(p, y0, seq(0, T, 1), f, "slade",
    eps_sw = .02, atol = 1e-9, rtol = 1e-9, maxsteps = 50000), silent = TRUE))))
  if (inherits(o, "try-error")) NULL else o }

NAS <- c("amp_long","amp_short","per_long","per_short","sd_dlnE","sd_dlnR","sd_dlnS","share_mean","share_2022","Rf9_max")
one <- function(row) {
  p <- mkp(tau_R = row$tau_R, tau_E = row$tau_E, n = row$n, eps_P = row$eps_P, k_R = row$k_R)
  y0 <- y0_share(p, row$sigma0)
  o <- raw_solve(p, y0, 61, funGDP); ok <- !is.null(o) && nrow(o) == 62 && all(is.finite(as.matrix(o)))
  if (ok) { lp <- log(o$Price); bl <- long_b(lp); bs <- short_b(lp); sh <- o$Recycling / (o$Extraction + o$Recycling)
    wm <- c(sd(bl), sd(bs), perf(bl), perf(bs), sd(diff(log(o$Extraction))), sd(diff(log(o$Recycling))),
            sd(diff(log(o$Extraction + o$Recycling))), mean(sh), tail(sh, 1), max(o$Recycling / o$f9))
    t_fail <- NA; rf9_fail <- NA
  } else { wm <- rep(NA, length(NAS))
    ff <- if (is.null(o)) NULL else o[apply(is.finite(as.matrix(o)), 1, all), ]
    t_fail <- if (is.null(ff) || !nrow(ff)) NA else max(ff$time) + 1961
    rf9_fail <- if (is.null(ff) || !nrow(ff)) NA else tail(ff$Recycling / ff$f9, 1) }
  names(wm) <- NAS
  oc <- raw_solve(p, y0, 139, fconst); okc <- !is.null(oc) && nrow(oc) == 140 && all(is.finite(as.matrix(oc)))
  pc <- if (okc) { lp <- log(oc$Price); bl <- long_b(lp); bs <- short_b(lp); e <- 1:60; l <- 81:140
    c(pers_long = sd(bl[l]) / sd(bl[e]), pers_short = sd(bs[l]) / sd(bs[e]),
      amp_long_c = sd(bl[e]), per_long_c = perf(bl[e]), per_short_c = perf(bs[e]))
  } else c(pers_long = NA, pers_short = NA, amp_long_c = NA, per_long_c = NA, per_short_c = NA)
  cbind(row, t(wm), t(pc), solved = ok, solved_const = okc, t_fail = t_fail, Rf9_at_fail = rf9_fail)
}

SIG <- c(0.05, 0.10, 0.15, round(0.92 / S0, 4), 0.25, 0.30, 0.35, 0.40)
G <- function(sig = SIG, tR = 2, tE = 8, n = FIT$n, e = FIT$eps_P, k = FIT$k_R, set)
  expand.grid(sigma0 = sig, tau_R = tR, tau_E = tE, n = n, eps_P = e, k_R = k, set = set, stringsAsFactors = FALSE)
grid <- rbind(G(tR = c(1, 2, 4, 8), tE = c(4, 8, 12, 16), set = "main"),
              G(n = 0.5, set = "n=0.5"), G(e = 0.2, set = "eps_P=0.2"),
              G(k = c(0.275, 0.353272), set = "k_R"),          # 0.353272 = k_E: equal gains
              G(n = 0.5, tR = c(1, 4, 8), set = "n=0.5 x tau_R"))
res <- do.call(rbind, mclapply(split(grid, seq_len(nrow(grid))), one, mc.cores = as.integer(Sys.getenv("D16_CORES", max(1L, parallel::detectCores() - 1L, na.rm = TRUE)))))
write.csv(res, file.path(OUT, "d16_sweep.csv"), row.names = FALSE)

# impulse diagnostic at the fit: 1% higher 1961 price, historical forcing to 2022, then flat GDP growth
p <- mkp(); y0 <- mky0(p); y1 <- y0; y1$P <- 1.01 * y0$P
ob <- solve1(p, y0, T = 139, tol = 1e-10); op <- solve1(p, y1, T = 139, tol = 1e-10)
imp <- data.frame(year = 1961 + ob$time, dlnP = (log(op$Price) - log(ob$Price)) / 0.01,
                  dlnE = (log(op$Extraction) - log(ob$Extraction)) / 0.01,
                  dlnR = (log(op$Recycling) - log(ob$Recycling)) / 0.01)
write.csv(imp, file.path(OUT, "d16_impulse.csv"), row.names = FALSE)
# trajectories for the figure: fitted delays, four shares, historical and constant forcing
tr <- do.call(rbind, lapply(c(0.05, 0.1876, 0.25), function(sig) {
  q <- mkp(); y <- y0_share(q, sig)
  a <- raw_solve(q, y, 61, funGDP); b <- raw_solve(q, y, 139, fconst)
  rbind(data.frame(forcing = "historical", sigma0 = sig, year = 1961 + a$time, P = a$Price, E = a$Extraction, R = a$Recycling, f9 = a$f9),
        data.frame(forcing = "constant", sigma0 = sig, year = 1961 + b$time, P = b$Price, E = b$Extraction, R = b$Recycling, f9 = b$f9)) }))
write.csv(tr, file.path(OUT, "d16_trajectories.csv"), row.names = FALSE)
cat("rows", nrow(res), "solved", sum(res$solved), "const", sum(res$solved_const), "\n")
