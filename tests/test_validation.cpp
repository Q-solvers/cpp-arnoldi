// Tests for constructor input validation of arnoldi::Arnoldi.
// Each test expects a specific std::invalid_argument to be thrown.

#include <arnoldi/arnoldi.hpp>

#include <cstdio>
#include <stdexcept>
#include <string>

#include <catch2/catch_test_macros.hpp>

// Catch2-backed expectation helpers. Signatures unchanged so the existing
// call sites need no edits; assertions are recorded against the running
// TEST_CASE. expect_throw requires a std::invalid_argument whose message
// contains `substr` (when non-empty).
template <typename F>
void expect_throw(const char* name, F&& f, const std::string& substr = "") {
    INFO(name);
    try {
        f();
        FAIL("no exception thrown");
    } catch (const std::invalid_argument& e) {
        INFO("message: " << e.what());
        if (!substr.empty())
            CHECK(std::string(e.what()).find(substr) != std::string::npos);
        else
            SUCCEED();
    } catch (const std::exception& e) {
        FAIL("unexpected exception: " << e.what());
    }
}

template <typename F>
void expect_no_throw(const char* name, F&& f) {
    INFO(name);
    try {
        f();
        SUCCEED();
    } catch (const std::exception& e) {
        FAIL("exception: " << e.what());
    }
}

TEST_CASE("test_n_zero", "[validation]") {
    expect_throw("Sym: n=0", []{ arnoldi::Arnoldi<arnoldi::Kind::Sym, double>("I", 0, "LM", 1, 3); }, "n must be");
}
TEST_CASE("test_n_negative", "[validation]") {
    expect_throw("Sym: n=-1", []{ arnoldi::Arnoldi<arnoldi::Kind::Sym, double>("I", -1, "LM", 1, 3); }, "n must be");
}

TEST_CASE("test_nev_zero", "[validation]") {
    expect_throw("Sym: nev=0", []{ arnoldi::Arnoldi<arnoldi::Kind::Sym, double>("I", 10, "LM", 0, 5); }, "nev must be");
}
TEST_CASE("test_nev_negative", "[validation]") {
    expect_throw("Nonsym: nev=-1", []{ arnoldi::Arnoldi<arnoldi::Kind::Nonsym, double>("I", 10, "LM", -1, 5); }, "nev must be");
}

TEST_CASE("test_ncv_too_large", "[validation]") {
    expect_throw("Sym: ncv > n", []{ arnoldi::Arnoldi<arnoldi::Kind::Sym, double>("I", 5, "LM", 2, 10); }, "ncv must be <= global");
}
TEST_CASE("test_ncv_sym_too_small", "[validation]") {
    expect_throw("Sym: ncv == nev", []{ arnoldi::Arnoldi<arnoldi::Kind::Sym, double>("I", 20, "LM", 5, 5); }, "ncv must satisfy");
}
TEST_CASE("test_ncv_nonsym_too_small", "[validation]") {
    expect_throw("Nonsym: ncv == nev+1", []{ arnoldi::Arnoldi<arnoldi::Kind::Nonsym, double>("I", 20, "LM", 4, 5); }, "ncv must satisfy");
}
TEST_CASE("test_ncv_nonsym_boundary", "[validation]") {
    expect_no_throw("Nonsym: ncv == nev+2 (valid)", []{ arnoldi::Arnoldi<arnoldi::Kind::Nonsym, double>("I", 20, "LM", 4, 6); });
}
TEST_CASE("test_ncv_sym_boundary", "[validation]") {
    expect_no_throw("Sym: ncv == nev+1 (valid)", []{ arnoldi::Arnoldi<arnoldi::Kind::Sym, double>("I", 20, "LM", 5, 6); });
}

TEST_CASE("test_bmat_invalid", "[validation]") {
    expect_throw("Sym: bmat='X'", []{ arnoldi::Arnoldi<arnoldi::Kind::Sym, double>("X", 10, "LM", 2, 5); }, "bmat must be");
}
TEST_CASE("test_bmat_empty", "[validation]") {
    expect_throw("Sym: bmat=''", []{ arnoldi::Arnoldi<arnoldi::Kind::Sym, double>("", 10, "LM", 2, 5); }, "bmat must be");
}
TEST_CASE("test_bmat_generalized", "[validation]") {
    expect_no_throw("Sym: bmat='G' (valid)", []{ arnoldi::Arnoldi<arnoldi::Kind::Sym, double>("G", 10, "LM", 2, 5); });
}

TEST_CASE("test_which_sym_invalid", "[validation]") {
    expect_throw("Sym: which='XX'", []{ arnoldi::Arnoldi<arnoldi::Kind::Sym, double>("I", 10, "XX", 2, 5); }, "which must be");
}
TEST_CASE("test_which_sym_LI", "[validation]") {
    // LI is valid for Nonsym but not Sym.
    expect_throw("Sym: which='LI'", []{ arnoldi::Arnoldi<arnoldi::Kind::Sym, double>("I", 10, "LI", 2, 5); }, "which must be");
}
TEST_CASE("test_which_sym_all_valid", "[validation]") {
    for (auto w : {"LM", "SM", "LA", "SA", "BE"}) {
        std::string name = std::string("Sym: which='") + w + "'";
        expect_no_throw(name.c_str(), [w]{ arnoldi::Arnoldi<arnoldi::Kind::Sym, double>("I", 20, w, 2, 5); });
    }
}

TEST_CASE("test_which_nonsym_invalid", "[validation]") {
    expect_throw("Nonsym: which='LA'", []{ arnoldi::Arnoldi<arnoldi::Kind::Nonsym, double>("I", 20, "LA", 2, 5); }, "which must be");
}
TEST_CASE("test_which_nonsym_all_valid", "[validation]") {
    for (auto w : {"LM", "SM", "LR", "SR", "LI", "SI"}) {
        std::string name = std::string("Nonsym: which='") + w + "'";
        expect_no_throw(name.c_str(), [w]{ arnoldi::Arnoldi<arnoldi::Kind::Nonsym, double>("I", 20, w, 2, 5); });
    }
}

TEST_CASE("test_which_herm_invalid", "[validation]") {
    using cplx = std::complex<double>;
    expect_throw("Herm: which='LI'", []{ arnoldi::Arnoldi<arnoldi::Kind::Herm, cplx>("I", 20, "LI", 2, 5); }, "which must be");
}
TEST_CASE("test_which_herm_all_valid", "[validation]") {
    using cplx = std::complex<double>;
    for (auto w : {"LM", "SM", "LA", "SA", "BE"}) {
        std::string name = std::string("Herm: which='") + w + "'";
        expect_no_throw(name.c_str(), [w]{ arnoldi::Arnoldi<arnoldi::Kind::Herm, cplx>("I", 20, w, 2, 5); });
    }
}
