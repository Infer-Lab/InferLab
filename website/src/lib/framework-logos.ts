// Logos for the frameworks and frontend backends the backend support matrix
// names, keyed by integration id or backend name. Provenance:
// src/assets/frameworks/SOURCES.md. A name without an entry is shown
// without a logo.
import vllm from '../assets/frameworks/vllm.png?url';
import sglang from '../assets/frameworks/sglang.svg?url';
import tokenspeed from '../assets/frameworks/tokenspeed.png?url';
import dynamo from '../assets/frameworks/dynamo.svg?url';

export const frameworkLogos: Record<string, string> = { vllm, sglang, tokenspeed };
export const backendLogos: Record<string, string> = { dynamo };
