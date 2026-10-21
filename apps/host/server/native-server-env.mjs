/** Let the private run config supply its key, without an inherited environment override. */
export function nativeServerEnv(inherited = process.env) {
  const env = { ...inherited, PYTHONUNBUFFERED: '1', PYTHONUTF8: '1', PYTHONIOENCODING: 'utf-8' };
  delete env.STRATA_API_KEY;
  return env;
}
