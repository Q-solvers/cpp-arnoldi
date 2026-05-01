// Tests for constructor input validation of arnoldi::Arnoldi.
// Each test expects a specific std::invalid_argument to be thrown.

#include <arnoldi/arnoldi.hpp>

#include <cstdio>
#include <stdexcept>
#include <string>

static int g_pass = 0, g_fail = 0;

template<typename F>
void expect_throw(const char* name, F&& f, const std::string& substr = "") {
    try {
        f();
        std::printf("  FAIL  %s — no exception thrown\n", name);
        ++g_fail;
    } catch (const std::invalid_argument& e) {
        if (!substr.empty() && std::string(e.what()).find(substr) == std::string::npos) {
            std::printf("  FAIL  %s — wrong message: %s\n", name, e.what());
            ++g_fail;
        } else {
            std::printf("  OK    %s\n", name);
            ++g_pass;
        }
    } catch (const std::exception& e) {
        std::printf("  FAIL  %s — unexpected exception: %s\n", name, e.what());
        ++g_fail;
    }
}

template<typename F>
void expect_no_throw(const char* name, F&& f) {
    try {
        f();
        std::printf("  OK    %s\n", name);
        ++g_pass;
    } catch (const std::exception& e) {
        std::printf("  FAIL  %s — exception: %s\n", name, e.what());
        ++g_fail;
    }
}

void test_n_zero() {
    expect_throw("Sym: n=0", []{ arnoldi::Arnoldi<arnoldi::Kind::Sym, double>("I", 0, "LM", 1, 3); }, "n must be");
}
void test_n_negative() {
    expect_throw("Sym: n=-1", []{ arnoldi::Arnoldi<arnoldi::Kind::Sym, double>("I", -1, "LM", 1, 3); }, "n must be");
}

void test_nev_zero() {
    expect_throw("Sym: nev=0", []{ arnoldi::Arnoldi<arnoldi::Kind::Sym, double>("I", 10, "LM", 0, 5); }, "nev must be");
}
void test_nev_negative() {
    expect_throw("Nonsym: nev=-1", []{ arnoldi::Arnoldi<arnoldi::Kind::Nonsym, double>("I", 10, "LM", -1, 5); }, "nev must be");
}

void test_ncv_too_large() {
    expect_throw("Sym: ncv > n", []{ arnoldi::Arnoldi<arnoldi::Kind::Sym, double>("I", 5, "LM", 2, 10); }, "ncv must be <= global");
}
void test_ncv_sym_too_small() {
    expect_throw("Sym: ncv == nev", []{ arnoldi::Arnoldi<arnoldi::Kind::Sym, double>("I", 20, "LM", 5, 5); }, "ncv must satisfy");
}
void test_ncv_nonsym_too_small() {
    expect_throw("Nonsym: ncv == nev+1", []{ arnoldi::Arnoldi<arnoldi::Kind::Nonsym, double>("I", 20, "LM", 4, 5); }, "ncv must satisfy");
}
void test_ncv_nonsym_boundary() {
    expect_no_throw("Nonsym: ncv == nev+2 (valid)", []{ arnoldi::Arnoldi<arnoldi::Kind::Nonsym, double>("I", 20, "LM", 4, 6); });
}
void test_ncv_sym_boundary() {
    expect_no_throw("Sym: ncv == nev+1 (valid)", []{ arnoldi::Arnoldi<arnoldi::Kind::Sym, double>("I", 20, "LM", 5, 6); });
}

void test_bmat_invalid() {
    expect_throw("Sym: bmat='X'", []{ arnoldi::Arnoldi<arnoldi::Kind::Sym, double>("X", 10, "LM", 2, 5); }, "bmat must be");
}
void test_bmat_empty() {
    expect_throw("Sym: bmat=''", []{ arnoldi::Arnoldi<arnoldi::Kind::Sym, double>("", 10, "LM", 2, 5); }, "bmat must be");
}
void test_bmat_generalized() {
    expect_no_throw("Sym: bmat='G' (valid)", []{ arnoldi::Arnoldi<arnoldi::Kind::Sym, double>("G", 10, "LM", 2, 5); });
}

void test_which_sym_invalid() {
    expect_throw("Sym: which='XX'", []{ arnoldi::Arnoldi<arnoldi::Kind::Sym, double>("I", 10, "XX", 2, 5); }, "which must be");
}
void test_which_sym_LI() {
    // LI is valid for Nonsym but not Sym.
    expect_throw("Sym: which='LI'", []{ arnoldi::Arnoldi<arnoldi::Kind::Sym, double>("I", 10, "LI", 2, 5); }, "which must be");
}
void test_which_sym_all_valid() {
    for (auto w : {"LM", "SM", "LA", "SA", "BE"}) {
        std::string name = std::string("Sym: which='") + w + "'";
        expect_no_throw(name.c_str(), [w]{ arnoldi::Arnoldi<arnoldi::Kind::Sym, double>("I", 20, w, 2, 5); });
    }
}

void test_which_nonsym_invalid() {
    expect_throw("Nonsym: which='LA'", []{ arnoldi::Arnoldi<arnoldi::Kind::Nonsym, double>("I", 20, "LA", 2, 5); }, "which must be");
}
void test_which_nonsym_all_valid() {
    for (auto w : {"LM", "SM", "LR", "SR", "LI", "SI"}) {
        std::string name = std::string("Nonsym: which='") + w + "'";
        expect_no_throw(name.c_str(), [w]{ arnoldi::Arnoldi<arnoldi::Kind::Nonsym, double>("I", 20, w, 2, 5); });
    }
}

void test_which_herm_invalid() {
    using cplx = std::complex<double>;
    expect_throw("Herm: which='LI'", []{ arnoldi::Arnoldi<arnoldi::Kind::Herm, cplx>("I", 20, "LI", 2, 5); }, "which must be");
}
void test_which_herm_all_valid() {
    using cplx = std::complex<double>;
    for (auto w : {"LM", "SM", "LA", "SA", "BE"}) {
        std::string name = std::string("Herm: which='") + w + "'";
        expect_no_throw(name.c_str(), [w]{ arnoldi::Arnoldi<arnoldi::Kind::Herm, cplx>("I", 20, w, 2, 5); });
    }
}

int main() {
    std::printf("test_validation:\n");

    test_n_zero();
    test_n_negative();
    test_nev_zero();
    test_nev_negative();
    test_ncv_too_large();
    test_ncv_sym_too_small();
    test_ncv_nonsym_too_small();
    test_ncv_nonsym_boundary();
    test_ncv_sym_boundary();
    test_bmat_invalid();
    test_bmat_empty();
    test_bmat_generalized();
    test_which_sym_invalid();
    test_which_sym_LI();
    test_which_sym_all_valid();
    test_which_nonsym_invalid();
    test_which_nonsym_all_valid();
    test_which_herm_invalid();
    test_which_herm_all_valid();

    std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
