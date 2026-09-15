export class ApiClient {
  constructor(baseUrl = location.origin) { this.baseUrl = baseUrl; }
  async request(path, {method = "GET", body, query} = {}) {
    const url = new URL("/api/v1" + path, this.baseUrl);
    for (const [key, value] of Object.entries(query || {})) url.searchParams.set(key, value);
    const response = await fetch(url, {method, headers: body ? {"Content-Type": "application/json"} : {},
      body: body ? JSON.stringify(body) : undefined});
    const data = await response.json();
    if (!response.ok) {
      const error = new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail || data));
      error.status = response.status; throw error;
    }
    return data;
  }
  async optional(path, query) {
    try { return await this.request(path, {query}); }
    catch (error) { if (error.status === 404) return null; throw error; }
  }
  async wait(jobId, onProgress) {
    while (true) {
      const job = await this.request("/jobs/" + jobId);
      onProgress(job);
      if (["succeeded", "failed"].includes(job.status)) return job;
      await new Promise(resolve => setTimeout(resolve, 700));
    }
  }
}
