import test from 'node:test';
import assert from 'node:assert/strict';
import { nativeServerEnv } from '../server/native-server-env.mjs';

test('source server gets its own config key, never an empty or inherited key override', () => {
  for (const value of ['', 'another-instance-secret']) {
    const inherited = { PATH: '/usr/bin', STRATA_API_KEY: value };
    const child = nativeServerEnv(inherited);
    assert.equal(Object.hasOwn(child, 'STRATA_API_KEY'), false);
    assert.equal(child.PATH, '/usr/bin');
    assert.equal(child.PYTHONUTF8, '1');
    assert.equal(inherited.STRATA_API_KEY, value);
  }
});
