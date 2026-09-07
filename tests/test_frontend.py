"""Run the shipped webpage script with a minimal DOM and mocked fetch in Node."""

from pathlib import Path
import shutil
import subprocess

import pytest


NODE = shutil.which("node") or ("/usr/bin/node" if Path("/usr/bin/node").is_file() else None)
PAGE = Path(__file__).resolve().parents[1] / "src/data_citadel/static/index.html"
HARNESS = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync(process.argv[1], 'utf8');
const scenario = process.argv[2];

function element(tag) {
  return {
    tag, value:'', textContent:'', children:[], events:{}, hidden:false,
    addEventListener(name, callback) { this.events[name] = callback; },
    replaceChildren(...children) {
      this.children = children;
      if(this.tag === 'select') this.value = children[0]?.value || '';
    },
    append(...children) { this.children.push(...children); }
  };
}
const nodes = Object.fromEntries([...html.matchAll(/<([a-z]+)[^>]*\bid="([^"]+)"[^>]*>/g)]
  .map(match => [match[2], element(match[1])]));
nodes.strategy.value = 'uniform';
const first = 'a'.repeat(32), second = 'b'.repeat(32);
const records = [
  {episode_id:first, action_id:'A_001', task_code:'first-task', instruction:'拿起苹果',
    reference_label:scenario === 'missing_reference' ? undefined : 'annotation_error'},
  {episode_id:second, action_id:'A_001', task_code:'second-task', instruction:'拿起杯子',
    reference_label:'correct'}
];
const experts = Array.from({length:5}, (_, index) => ({
  episode_id:`expert-${index}`, instruction:`专家动作 ${index + 1}`
}));
const result = {
  episode_id:first, action_id:'A_001', verdict:'correct', ground_truth_candidate:true,
  error_types:[], reason:'模型确认动作已完成',
  provenance:{
    expert_ids:experts.map(item => item.episode_id), expert_tasks:experts,
    expert_timestamps_s:Object.fromEntries(experts.map(item => [item.episode_id, [0, 1, 2]]))
  },
  assessments:{
    generic:{verdict:'correct', findings:[], reason:'画面质量通过'},
    task:{verdict:'correct', findings:[], reason:'看到了抓取过程和完成状态'}
  }
};
if(scenario === 'incorrect_labels') {
  result.verdict = 'incorrect';
  result.ground_truth_candidate = false;
  result.error_types = ['blurred', 'incomplete_action'];
  result.assessments.generic = {verdict:'incorrect', findings:[{code:'blurred'}, {code:'other'}], reason:'画面模糊'};
  result.assessments.task = {verdict:'incorrect', findings:[{code:'incomplete_action'}], reason:'动作被截断'};
}else if(scenario === 'repeated_retry') {
  result.verdict = 'incorrect'; result.ground_truth_candidate = false;
  result.error_types = ['repeated_retry'];
  result.assessments.task = {verdict:'incorrect', findings:[{code:'repeated_retry'}], reason:'重试后成功'};
}else if(scenario === 'strict_uncertain') {
  result.verdict = 'uncertain';
  result.ground_truth_candidate = false;
  result.reason = '严格审核证据不足，原始 Qwen 分项通过';
}
const requests = [];
let finishReview;
const response = value => ({ok:true, json:async () => value});
const context = {
  document:{getElementById:id => nodes[id], createElement:element},
  Option:function(text, value) { this.text = text; this.value = value; },
  fetch:async (path, options) => {
    if(path === '/v1/episodes') return response(records);
    assert.equal(path, '/v1/reviews');
    requests.push(JSON.parse(options.body));
    return new Promise(resolve => { finishReview = () => resolve(response(result)); });
  }
};

(async () => {
  vm.runInNewContext(html.match(/<script>([\s\S]*?)<\/script>/)[1], context);
  await new Promise(setImmediate); // Complete the initial episode-list request.
  const sourceText = scenario === 'missing_reference' ? '未提供' : '标注信息错误（annotation_error）';
  assert.ok(nodes.groundTruth.textContent.startsWith('GT（Ground Truth，原人工审核标签）：'));
  assert.ok(nodes.groundTruth.textContent.includes(sourceText));
  const pending = nodes.review.events.click();
  assert.deepEqual(requests, [{episode_id:first, strategy:'uniform'}]);
  if(scenario === 'selection_during_request') {
    nodes.episode.value = second;
    nodes.episode.events.change();
    assert.ok(nodes.groundTruth.textContent.includes('正确（correct）'));
  }
  finishReview();
  await pending;
  assert.equal(nodes.result.hidden, false);
  assert.ok(nodes.resultGroundTruth.textContent.startsWith('GT（Ground Truth，原人工审核标签）：'));
  assert.ok(nodes.resultGroundTruth.textContent.includes(sourceText));
  assert.ok(html.indexOf('id="resultGroundTruth"') < html.indexOf('id="decision"'));
  const expectedLabel = {
    incorrect_labels:'画面模糊（blurred）', repeated_retry:'反复重试（repeated_retry）',
    strict_uncertain:'无法确认（uncertain）'
  }[scenario] || '正确（correct）';
  assert.equal(nodes.decision.textContent, 'Predict（最终审核标签）：' + expectedLabel);
  assert.deepEqual(JSON.parse(nodes.details.textContent).error_types, result.error_types);
  const rawPanel = [...html.matchAll(/<details\b([^>]*)>([\s\S]*?)<\/details>/g)]
    .find(match => match[2].includes('id="qwenLabels"'));
  assert.ok(rawPanel, 'Original Qwen labels must be inside an expandable details panel');
  assert.doesNotMatch(rawPanel[1], /\bopen(?:\s|=|$)/);
  const rawLabels = nodes.qwenLabels.textContent;
  assert.ok(rawLabels.includes('通用') && rawLabels.includes('任务'));
  if(scenario === 'incorrect_labels') {
    assert.equal(rawLabels, 'Qwen 原始标签：通用 画面模糊（blurred）；任务 动作不完整（incomplete_action）');
  }else if(scenario === 'repeated_retry') {
    assert.equal(rawLabels, 'Qwen 原始标签：通用 正确（correct）；任务 反复重试（repeated_retry）');
  }else{
    assert.equal((rawLabels.match(/正确（correct）/g) || []).length, 2);
  }
  assert.equal(nodes.reason.textContent, result.reason);
  assert.ok(nodes.expertNote.textContent.includes('5/5 条'));
  assert.equal(nodes.expertList.children.length, 5);
  for(let index = 0; index < 5; index++) {
    assert.ok(nodes.expertList.children[index].textContent.includes(experts[index].instruction));
    assert.ok(nodes.expertList.children[index].textContent.includes('3 帧'));
  }
  assert.ok(nodes.taskAssessment.textContent.includes(result.assessments.task.reason));
  assert.equal(nodes.status.textContent, '完成');
  if(scenario === 'incorrect_labels') {
    result.error_types.reverse();
    const reordered = nodes.review.events.click();
    finishReview(); await reordered;
    assert.equal(nodes.decision.textContent, 'Predict（最终审核标签）：动作不完整（incomplete_action）');
    assert.deepEqual(JSON.parse(nodes.details.textContent).error_types, ['incomplete_action', 'blurred']);
  }
  if(scenario === 'missing_reference') {
    result.verdict = 'uncertain'; result.ground_truth_candidate = false;
    result.provenance = {expert_ids:[], expert_tasks:[]};
    result.assessments.task = {verdict:'uncertain', reason:'没有可用专家'};
    const unavailable = nodes.review.events.click();
    finishReview(); await unavailable;
    assert.match(nodes.qwenLabels.textContent, /任务.*未执行/);
    assert.ok(nodes.taskAssessment.textContent.includes('未调用 Qwen'));
    assert.ok(nodes.resultGroundTruth.textContent.includes('未提供'));
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
"""


@pytest.mark.skipif(NODE is None, reason="Node is unavailable; no frontend dependency is installed")
@pytest.mark.parametrize("scenario", [
    "separate_labels", "missing_reference", "selection_during_request",
    "incorrect_labels", "strict_uncertain", "repeated_retry",
])
def test_ground_truth_and_one_prediction_display_actual_labels(scenario):
    completed = subprocess.run(
        [NODE, "-e", HARNESS, str(PAGE), scenario], capture_output=True, text=True, timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
