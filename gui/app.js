import {ApiClient} from "./api.js";
import {$,state,labels,outcome} from "./state.js";
import {showMedia,showResult} from "./views.js";

const api = new ApiClient();
const stages = {queued:"排队",configuration:"核对配置",source:"检查数据",resources:"读取参考资源",
  media:"准备视频和信号",model:"任务审核",quality:"画质审核",evidence:"核验证据",done:"完成"};
let selection = 0, history = [];
const query = () => ({run_id:state.runId});
const reportError = error => $("notice").textContent = error.message;
const format = value => JSON.stringify(value, null, 2);
function busy(value) {
  for (const id of ["run","filter","episode","prepare","review"]) $(id).disabled = value;
}
function matches(row, filter) {
  const kind = outcome(row);
  return filter === "all" || (filter === "errors" ? ["false_accept","false_reject"].includes(kind) : kind === filter);
}
async function stats() {
  const reports = await Promise.all(state.splits.map(split => api.request("/reports",{query:{...query(),split}})));
  const keys = ["total","completed","needs_review","failed","not_run","false_accept","false_reject"];
  const n = Object.fromEntries(keys.map(key => [key,reports.reduce((sum,r)=>sum+r.counts[key],0)]));
  $("stats").textContent = "共 "+n.total+" 条 · 已判定 "+n.completed+" · 待复核 "+n.needs_review+
    " · 处理失败 "+n.failed+" · 未审核 "+n.not_run+" · 误放 "+n.false_accept+" · 误拒 "+n.false_reject;
}
async function loadHistory(id, token) {
  const values = await api.request("/episodes/"+id+"/history",{query:query()});
  if (token !== selection) return;
  history = values;
  $("history").replaceChildren();
  for (const result of history) {
    const option = document.createElement("option");
    option.value = result.result_id;
    option.textContent = result.created_at+" · "+result.configuration_sha256.slice(0,12)+" · "+
      (labels[result.label || result.status] || result.status);
    $("history").append(option);
  }
  $("compare").disabled = !history.length;
}
async function selectEpisode() {
  const token = ++selection, id = $("episode").value;
  state.result = null; state.media = null;
  for (const name of ["result","videos","signals","debug"]) $(name).hidden = true;
  for (const name of ["preview","trace","current-version","past-version"]) $(name).textContent = "";
  for (const video of document.querySelectorAll("video")) {video.pause();video.removeAttribute("src");video.load();}
  $("review").textContent = "开始审核";
  if (!id) { $("notice").textContent = "当前没有符合筛选的记录。"; return; }
  const url = new URL(location.href);
  url.searchParams.set("run",state.runId);url.searchParams.set("episode",id);
  url.searchParams.set("filter",$("filter").value);window.history.replaceState(null,"",url);
  const row = state.rows.find(row=>row.episode_id === id);
  const [media,result] = await Promise.all([
    api.optional("/episodes/"+id+"/media",query()),
    row?.result_id ? api.request("/results/"+row.result_id,{query:query()}) : null]);
  if (token !== selection) return;
  if (media) showMedia(media);
  if (result) await displayResult(result);
  $("debug").hidden = false;
  await loadHistory(id,token);
  $("notice").textContent = media ? "视频已加载。输入预览不会调用模型。" : "先准备输入预览，检查视频和 prompt。";
}
async function displayResult(result) {
  showResult(result);
  $("debug").hidden = false;
  $("version").textContent = "本次配置："+result.configuration_sha256;
  $("current-version").textContent = format(result);
  if (result.result_id) $("trace").textContent = format(await api.request("/results/"+result.result_id+"/trace",{query:query()}));
}
async function filterRows(preferred) {
  $("episode").replaceChildren();
  for (const row of state.rows.filter(row=>matches(row,$("filter").value))) {
    const option = document.createElement("option");option.value = row.episode_id;
    option.textContent = row.task_code+" · "+row.episode_id.slice(0,10)+" · "+
      (labels[outcome(row)] || labels[row.label || row.status])+" · "+(row.split === "holdout" ? "留出集" : "开发集");
    $("episode").append(option);
  }
  if ([...$("episode").options].some(option=>option.value === preferred)) $("episode").value = preferred;
  await selectEpisode();
}
async function loadRun() {
  state.runId = $("run").value;state.splits = ["development"];
  const [rows,settings] = await Promise.all([
    api.request("/episodes",{query:query()}),api.request("/settings",{query:query()})]);
  state.rows = rows;
  try {
    const holdout = await api.request("/episodes",{query:{...query(),split:"holdout"}});
    state.rows.push(...holdout);state.splits.push("holdout");
  } catch(error) {if(error.status !== 409)throw error;}
  $("task").textContent = "模型："+settings.model.model+" · 配置 "+settings.configuration_sha256.slice(0,12);
  const params = new URLSearchParams(location.search);
  if ([...$("filter").options].some(o=>o.value === params.get("filter"))) $("filter").value = params.get("filter");
  await filterRows(params.get("episode"));await stats();
}
async function follow(jobId) {
  busy(true);
  try {
    const job = await api.wait(jobId, value => {
      const seconds = (Date.now() - value.created_ns / 1e6) / 1000;
      $("notice").textContent = (stages[value.stage] || value.stage)+" · "+seconds.toFixed(1)+" 秒 · 任务 "+jobId.slice(0,10);
    });
    if (job.status === "failed" && !job.result_id) throw new Error(job.error?.message || job.error?.type || "任务失败");
    const value = await api.request("/jobs/"+jobId+"/output");
    $("debug").hidden = false;
    if (job.kind === "review") await displayResult(value);
    else {
      $("preview").textContent = format(value);
      $("version").textContent = "输入配置："+value.configuration_sha256;
    }
    const media = await api.optional("/episodes/"+job.episode_id+"/media",query());
    if(media)showMedia(media);
    await loadHistory(job.episode_id,selection);await stats();
    $("notice").textContent = job.status === "failed" ? "处理失败，可查看原因并显式重试。" : "任务完成。";
  } finally {
    busy(false);
    const url = new URL(location.href);url.searchParams.delete("job");window.history.replaceState(null,"",url);
  }
}
async function action(preview) {
  busy(true);
  try {
    const job = await api.request(preview ? "/previews" : "/reviews",{method:"POST",
      body:{run_id:state.runId,episode_id:$("episode").value,retry_failed:true}});
    const url = new URL(location.href);url.searchParams.set("job",job.job_id);window.history.replaceState(null,"",url);
    await follow(job.job_id);
  } catch(error) {reportError(error);}
  finally {busy(false);}
}
$("run").onchange = () => loadRun().catch(reportError);
$("filter").onchange = () => filterRows($("episode").value).catch(reportError);
$("episode").onchange = () => selectEpisode().catch(reportError);
$("prepare").onclick = () => action(true);
$("review").onclick = () => action(false);
$("compare").onclick = () => {
  $("current-version").textContent = state.result ? format(state.result) : "当前配置尚未审核";
  $("past-version").textContent = format(history.find(value=>value.result_id === $("history").value));
};
(async()=>{
  const runs = await api.request("/runs"), params = new URLSearchParams(location.search);
  for(const run of runs) {
    const option = document.createElement("option");option.value=run.run_id;
    option.textContent=run.run_id+" · "+run.episodes+" 条";$("run").append(option);
  }
  if(!runs.length){$("notice").textContent="请在 config/server.toml 配置数据集后重启服务。";busy(true);return;}
  if(runs.some(run=>run.run_id===params.get("run")))$("run").value=params.get("run");
  await loadRun();
  if(params.get("job"))await follow(params.get("job"));
})().catch(reportError);
