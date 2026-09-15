import {$,state,names,labels,outcome} from "./state.js";
export function showMedia(media){
  state.media=media;$("videos").hidden=false;showGripper(media);
  for(const name of ["stitched","main","left_wrist","right_wrist"]){
    const item=media.videos[name];$(name).hidden=!item;
    if(item)$(name).src=item.url;else $(name).removeAttribute("src");
    const link=$("download_"+name);
    if(link&&item){link.href=item.url;link.download=$("episode").value+"_"+name+".mp4"}
  }
}
export function showGripper(media, cursor=null){
  const channels=media.gripper?.channels||{}, entries=Object.entries(channels).filter(([,c])=>c.samples.length);
  $("signals").hidden=!entries.length;if(!entries.length)return;
  const canvas=$("gripper"),ctx=canvas.getContext("2d"),left=70,right=1180;
  const end=Math.max(...entries.map(([,c])=>c.samples.at(-1)[0]));if(end<=0)return;
  ctx.clearRect(0,0,canvas.width,canvas.height);ctx.font="14px sans-serif";
  const legend={left_position:["左夹爪","#7fb8ff"],right_position:["右夹爪","#ffbd71"],
    left_finger0:["左指0","#7fb8ff"],left_finger1:["左指1","#ba9bff"],
    right_finger0:["右指0","#ffbd71"],right_finger1:["右指1","#ed8ba6"]};
  for(const [row,isPosition] of [true,false].entries()){
    const series=entries.filter(([name])=>name.endsWith("_position")===isPosition),top=30+row*170,bottom=top+120;
    const values=series.flatMap(([,c])=>c.samples.map(p=>p[1]));
    const low=values.reduce((a,b)=>Math.min(a,b),0),high=values.reduce((a,b)=>Math.max(a,b),1);
    const x=t=>left+t/end*(right-left),y=v=>bottom-(v-low)/(high-low)*(bottom-top);
    ctx.fillStyle="#c6d1db";ctx.fillText(isPosition?"夹爪位置（原始值）":"触觉相对强度（未标定）",left,top-12);
    ctx.strokeStyle="#374653";ctx.beginPath();ctx.moveTo(left,top);ctx.lineTo(left,bottom);ctx.lineTo(right,bottom);ctx.stroke();
    ctx.fillText(high.toFixed(1),5,top+5);ctx.fillText(low.toFixed(1),5,bottom);
    series.forEach(([name,channel],i)=>{
      const [label,color]=legend[name];ctx.strokeStyle=color;ctx.beginPath();
      channel.samples.forEach(([t,v],j)=>j&&t-channel.samples[j-1][0]<=0.1?ctx.lineTo(x(t),y(v)):ctx.moveTo(x(t),y(v)));ctx.stroke();
      ctx.fillStyle=color;ctx.fillText(label,650+i*120,top-12);
    });
    for(let i=0;i<=5;i++){ctx.fillStyle="#9eafbf";ctx.fillText((end*i/5).toFixed(1)+"s",x(end*i/5)-10,bottom+20)}
    if(cursor!==null){ctx.strokeStyle="#83d8ab";ctx.beginPath();ctx.moveTo(x(cursor),top);ctx.lineTo(x(cursor),bottom);ctx.stroke()}
  }
  canvas.onclick=event=>{
    const rect=canvas.getBoundingClientRect(),x=(event.clientX-rect.left)*canvas.width/rect.width;
    const time=Math.max(0,Math.min(end,(x-left)/(right-left)*end));
    for(const [name,item] of Object.entries(media.videos)){
      const video=$(name);if(video?.readyState)video.currentTime=Math.max(0,Math.min(video.duration,time-item.start_s));
    }
    showGripper(media,time);
  };
}
export function showResult(result){
  state.result=result;$("result").hidden=false;
  $("conclusion").textContent=labels[result.label||result.status]||result.status;
  $("reason").textContent=result.reason||"";
  const row=state.rows.find(row=>row.episode_id===result.episode_id);
  if(row)Object.assign(row,{label:result.label,status:result.status,result_id:result.result_id});
  const kind=outcome(row||{});
  $("comparison").textContent=(labels[kind]?labels[kind]+" · ":"")+"GT："+(labels[result.evaluation?.gt]||"无标签")+
    (result.evaluation?.gt_reason?" · GT原因："+result.evaluation.gt_reason:"");
  $("comparison").className=kind==="false_accept"||kind==="false_reject"?"fail":"muted";
  $("review").textContent=result.status==="failed"?"重试失败任务":"查看已有结果";
  $("checks").replaceChildren();$("observations").replaceChildren();$("evidence").replaceChildren();
  const evidence=new Map((result.evidence||[]).map(frame=>[frame.frame_id,frame]));
  const cameraNames={main:"主镜头画质",left_wrist:"左腕画质",right_wrist:"右腕画质"};
  for(const [name,check] of Object.entries({...result.checks,...result.quality_by_camera,...(result.hold?{hold:result.hold}:{})})){
    const row=document.createElement("tr");
    for(const value of [cameraNames[name]||names[name]||name,labels[check.state],check.evidence_ids.map(id=>evidence.get(id)?.time_s.toFixed(2)+"s").join("、")+(check.description?" · "+check.description:"")]){
      const cell=document.createElement("td");cell.textContent=value;row.append(cell);
    }
    row.children[1].className=check.state;$("checks").append(row);
  }
  for(const item of result.observations||[]){
    const li=document.createElement("li");li.textContent=item.description;$("observations").append(li);
  }
  const visibility=result.main_visibility_by_frame||{};
  $("main-audit").hidden=!Object.keys(visibility).length;
  $("main-audit").open=Object.values(visibility).includes("absent");
  $("main-frames").replaceChildren();
  const states={visible:["可见","pass"],absent:["出画","fail"],uncertain:["待确认","unknown"],no_frame:["无有效帧","muted"]};
  for(const [id,status] of Object.entries(visibility)){
    const frame=evidence.get(id);if(!frame)continue;
    const button=document.createElement("button"),[label,color]=states[status];
    button.textContent=frame.time_s.toFixed(2)+"s "+label;button.className=color;
    button.dataset.frame=id;button.onclick=()=>$("frame-"+id)?.click();$("main-frames").append(button);
  }
  for(const frame of result.evidence||[]){
    const figure=document.createElement("figure"),img=document.createElement("img"),caption=document.createElement("figcaption");
    img.src=frame.url;
    img.id="frame-"+frame.frame_id;img.alt=frame.time_s.toFixed(2)+"秒三路证据帧";
    img.onclick=()=>{
      if(!state.media)return;
      for(const [name,item] of Object.entries(state.media.videos)){
        const video=$(name);if(!video?.readyState)continue;
        const time=name==="stitched"?frame.video_time_s:frame.time_s-item.start_s;
        video.currentTime=Math.max(0,Math.min(video.duration,time||0));
      }
      showGripper(state.media,frame.time_s);
    };
    caption.textContent=frame.time_s.toFixed(2)+" 秒";figure.append(img,caption);$("evidence").append(figure);
  }
}
