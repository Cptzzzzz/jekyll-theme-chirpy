/*
 * Deliberately incomplete test double used only by the chapter 1 loader probe.
 * It demonstrates ELF search-path precedence; it is not a usable NCCL library.
 */
int ncclGetVersion(int* version) {
  if (version == 0) return 4;
  *version = 29999; /* Decodes as the intentionally impossible 2.99.99. */
  return 0;
}
