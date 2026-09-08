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
    tag, value:'', textContent:'', children:[], events:{}, dataset:{}, hidden:false,
    readyState:0, currentTime:0, paused:true, loadCalls:0,
    pause() { this.paused = true; },
    load() { this.loadCalls++; this.readyState = 0; },
    removeAttribute(name) { delete this[name]; },
    querySelectorAll(tag) {
      return this.children.flatMap(child => [
        ...(child.tag === tag ? [child] : []), ...(child.querySelectorAll?.(tag) || [])
      ]);
    },
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
nodes.reviewMode.value = 'main';
assert.match(html, /id="reviewMode"[^>]*>\s*<option value="main"/);
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
const requests = [], previewRequests = [];
let finishReview, finishPreview;
const previewResult = {
  duration_s:5, strategy:scenario === 'preview_keyframes' ? 'keyframes' : 'uniform', warnings:[], motion:{},
  videos:Object.fromEntries(Object.entries({main:0.25, left_wrist:2.25, right_wrist:0.5}).map(([view, start_s]) => [
    view, {url:'/v1/episodes/' + first + '/video?view=' + view, start_s}
  ])),
  frames:[['main', 1.25], ['left_wrist', 3.75], ['left_wrist', 1.0], ['right_wrist', 3.0]]
    .map(([view, timestamp_s]) => ({view, timestamp_s, image:'data:image/jpeg;base64,' + Buffer.from(view).toString('base64')}))
};
if(scenario === 'preview_missing_view') {
  delete previewResult.videos.right_wrist;
  previewResult.frames = previewResult.frames.filter(frame => frame.view !== 'right_wrist');
}
const response = value => ({ok:true, json:async () => value});
const context = {
  document:{getElementById:id => nodes[id], createElement:element},
  Option:function(text, value) { this.text = text; this.value = value; },
  fetch:async (path, options) => {
    if(path === '/v1/episodes') return response(records);
    if(path.includes('/frames?')) {
      previewRequests.push(path);
      return new Promise(resolve => {
        finishPreview = () => resolve(scenario === 'preview_stale_error'
          ? {ok:false, json:async () => ({detail:'旧预览失败'})} : response(previewResult));
      });
    }
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
  if(scenario.startsWith('preview_')) {
    nodes.strategy.value = previewResult.strategy;
    const pending = nodes.preview.events.click();
    assert.deepEqual(previewRequests, [
      '/v1/episodes/' + first + '/frames?strategy=' + previewResult.strategy + '&interval_s=2&camera_mode=main_wrist'
    ]);
    assert.equal(requests.length, 0);
    if(scenario === 'preview_stale_response' || scenario === 'preview_stale_error') {
      nodes.episode.value = second;
      nodes.episode.events.change();
      nodes.status.textContent = '已选择新记录';
      finishPreview(); await pending;
      assert.equal(nodes.status.textContent, '已选择新记录');
      assert.equal(nodes.frames.children.length, 0);
      assert.equal(nodes.previewCard.hidden, true);
      assert.ok(nodes.groundTruth.textContent.includes('正确（correct）'));
      assert.equal(nodes.preview.disabled, false);
      return;
    }
    finishPreview(); await pending;
    assert.equal(nodes.previewCard.hidden, false);
    assert.equal(nodes.frames.children.length, 3);
    assert.deepEqual(nodes.frames.children.map(group => group.dataset.view), ['main', 'left_wrist', 'right_wrist']);
    assert.deepEqual(nodes.frames.children.map(group => group.children[0].textContent), ['主视角', '左腕 Wrist', '右腕 Wrist']);
    if(previewResult.strategy === 'uniform') {
      assert.match(nodes.samplingNote.textContent, /均匀抽帧/);
      assert.doesNotMatch(nodes.samplingNote.textContent, /事件/);
    }else{
      assert.match(nodes.samplingNote.textContent, /事件优先.*回退均匀/);
    }
    for(const group of nodes.frames.children) {
      const view = group.dataset.view, source = previewResult.videos[view];
      const videos = group.querySelectorAll('video');
      const frames = previewResult.frames.filter(frame => frame.view === view);
      assert.deepEqual(group.querySelectorAll('img').map(img => img.src), frames.map(frame => frame.image));
      if(!source) {
        assert.equal(videos.length, 0);
        assert.equal(group.querySelectorAll('button').length, 0);
        assert.ok(group.children.some(child => child.textContent.includes('不可用')));
        continue;
      }
      assert.equal(videos.length, 1);
      const video = videos[0];
      assert.equal(video.dataset.view, view);
      assert.equal(video.src, source.url);
      assert.equal(video.controls, true);
      assert.equal(video.preload, 'none');
      assert.ok(video.title.endsWith('对应视频'));
      const jump = group.querySelectorAll('button')[0];
      assert.equal(jump.disabled, false);
      video.paused = false;
      jump.events.click();
      assert.equal(video.currentTime, 0, 'Seek waits for this video metadata');
      assert.equal(video.paused, true);
      assert.equal(video.loadCalls, 1);
      assert.equal(typeof video.onloadedmetadata, 'function');
      video.readyState = 1;
      video.onloadedmetadata();
      assert.equal(video.currentTime, Math.max(0, frames[0].timestamp_s - source.start_s));
      assert.equal(video.paused, true);
      assert.equal(video.onloadedmetadata, null);
    }
    const left = nodes.frames.children[1], leftVideo = left.querySelectorAll('video')[0];
    leftVideo.paused = false;
    left.querySelectorAll('button')[1].events.click();
    assert.equal(leftVideo.currentTime, 0, 'Before-source timestamps clamp to zero');
    assert.equal(leftVideo.loadCalls, 1, 'Loaded metadata permits direct seek without reload');
    assert.equal(leftVideo.paused, true);
    if(scenario === 'preview_video_error') {
      const failure = left.children.find(child => child.textContent.includes('载入失败'));
      assert.ok(failure && failure.hidden);
      leftVideo.events.error();
      assert.equal(failure.hidden, false);
      assert.match(failure.textContent, /左腕 Wrist.*对应视频载入失败/);
      leftVideo.events.error();
      assert.equal(left.children.filter(child => !child.hidden && child.textContent.includes('载入失败')).length, 1);
      for(const group of [nodes.frames.children[0], nodes.frames.children[2]]) {
        assert.ok(group.children.every(child => child.hidden || !child.textContent.includes('载入失败')));
      }
      assert.equal(nodes.status.textContent, '完成');
    }
    const oldVideos = nodes.frames.querySelectorAll('video');
    oldVideos.forEach(video => { video.paused = false; });
    oldVideos[0].readyState = 0;
    nodes.frames.children[0].querySelectorAll('button')[0].events.click();
    assert.equal(typeof oldVideos[0].onloadedmetadata, 'function');
    nodes.episode.value = second;
    nodes.episode.events.change();
    assert.equal(nodes.previewCard.hidden, true);
    assert.equal(nodes.frames.children.length, 0);
    for(const video of oldVideos) {
      assert.equal(video.paused, true);
      assert.equal(video.src, undefined);
      assert.equal(video.onloadedmetadata, null);
      assert.ok(video.loadCalls >= 2);
    }
    return;
  }
  if(scenario === 'review_multiview') nodes.reviewMode.value = 'main_wrist';
  const pending = nodes.review.events.click();
  assert.deepEqual(requests, [{
    episode_id:first, strategy:'uniform', camera_mode:scenario === 'review_multiview' ? 'main_wrist' : 'main'
  }]);
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
    "incorrect_labels", "strict_uncertain", "repeated_retry", "review_multiview",
    "preview_three_views", "preview_keyframes", "preview_missing_view", "preview_stale_response",
    "preview_video_error", "preview_stale_error",
])
def test_frontend_preserves_labels_and_routes_camera_previews(scenario):
    completed = subprocess.run(
        [NODE, "-e", HARNESS, str(PAGE), scenario], capture_output=True, text=True, timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
