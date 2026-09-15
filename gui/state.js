export const $=id=>document.getElementById(id);
export const state={runId:null,rows:[],splits:["development"],result:null,media:null};
export const names={object_match:"物体一致",scene_match:"场景一致",main_visibility:"主镜头可见性",image_quality:"画面质量",action:"任务动作",retry_free:"无失败重试",completeness:"动作完整性",hold:"持续悬空"};
export const labels={false_accept:"误放",false_reject:"误拒",correct:"通过",incorrect:"不通过",needs_review:"待复核",failed:"处理失败",not_run:"未审核",completed:"已审核",pass:"符合",fail:"不符合",unknown:"待确认"};

export function outcome(row){
  if(row.gt==="incorrect"&&row.label==="correct")return "false_accept";
  if(row.gt==="correct"&&row.label==="incorrect")return "false_reject";
  return row.status==="needs_review"?"needs_review":null;
}

export function episodeLabel(row){
  return row.task_code+" · "+row.episode_id.slice(0,10)+" · "+
    (labels[outcome(row)] || labels[row.label || row.status])+" · "+
    (row.split === "holdout" ? "留出集" : "开发集");
}
