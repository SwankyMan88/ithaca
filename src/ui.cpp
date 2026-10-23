// Embedded single-page UI (no external assets: works fully offline).
extern const char* kIndexHtml;
const char* kIndexHtml = R"HTML(<!doctype html>
<html lang="en" data-theme="dark"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ithaca</title>
<style>
:root{--bg:#0f0f11;--side:#161618;--panel:#1c1c1f;--raise:#242428;--line:#2c2c31;--text:#ececf1;--dim:#9a9aa6;--faint:#6b6b76;
--a1:#8b6cff;--a2:#3fb6ff;--grad:linear-gradient(135deg,var(--a1),var(--a2));--user:#26262b;--ok:#3fb950;--warn:#e3a33b;--bad:#f0605a;
--gpu:#3fb950;--ram:#58a6ff;--disk:#5c5c66;--code:#121214;--shadow:0 10px 40px rgba(0,0,0,.45)}
[data-theme=light]{--bg:#ffffff;--side:#f6f6f8;--panel:#ffffff;--raise:#efeff3;--line:#e3e3e8;--text:#1d1d22;--dim:#5f5f6b;--faint:#9a9aa6;--user:#f0f0f4;--code:#f6f6f9;--shadow:0 10px 40px rgba(0,0,0,.12)}
*{box-sizing:border-box}html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.6 "Segoe UI Variable Text","Segoe UI",Inter,system-ui,sans-serif;display:flex;overflow:hidden}
button{font:inherit;color:inherit;background:none;border:0;cursor:pointer}input,textarea,select{font:inherit;color:var(--text)}
.icon{width:18px;height:18px;stroke:currentColor;fill:none;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round;flex-shrink:0}
/* sidebar */
aside{width:268px;background:var(--side);border-right:1px solid var(--line);display:flex;flex-direction:column;flex-shrink:0;transition:margin .2s}
aside.hidden{margin-left:-268px}
.brand{display:flex;align-items:center;gap:10px;padding:16px 16px 10px;font-weight:650;font-size:17px;letter-spacing:.3px}
.logo{width:26px;height:26px;border-radius:8px;background:var(--grad);display:grid;place-items:center;box-shadow:0 0 18px rgba(139,108,255,.45)}
.logo svg{width:16px;height:16px;fill:#fff}
.newchat{margin:6px 12px 10px;display:flex;align-items:center;gap:8px;padding:9px 12px;border:1px solid var(--line);border-radius:10px;background:var(--panel)}
.newchat:hover{background:var(--raise)}
.search{margin:0 12px 8px;padding:7px 10px;border-radius:9px;border:1px solid var(--line);background:transparent;width:calc(100% - 24px)}
#history{flex:1;overflow:auto;padding:0 8px}
.group{font-size:11.5px;color:var(--faint);padding:12px 8px 4px;font-weight:600;text-transform:uppercase;letter-spacing:.6px}
.hitem{display:flex;align-items:center;gap:6px;padding:7px 9px;border-radius:8px;cursor:pointer;color:var(--dim);font-size:14px}
.hitem span{flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.hitem:hover,.hitem.on{background:var(--raise);color:var(--text)}
.hitem .x{opacity:0;padding:2px;border-radius:5px}.hitem:hover .x{opacity:.7}.hitem .x:hover{opacity:1;background:var(--line)}
.sidefoot{border-top:1px solid var(--line);padding:10px 12px;display:flex;align-items:center;gap:10px}
.sidefoot button{display:flex;align-items:center;gap:9px;padding:8px 10px;border-radius:9px;flex:1}
.sidefoot button:hover{background:var(--raise)}
/* main */
main{flex:1;display:flex;flex-direction:column;min-width:0;position:relative}
.topbar{height:54px;display:flex;align-items:center;gap:10px;padding:0 14px;border-bottom:1px solid var(--line)}
.tbtn{padding:7px;border-radius:8px;color:var(--dim)}.tbtn:hover{background:var(--raise);color:var(--text)}
.modelpick{display:flex;align-items:center;gap:6px;padding:6px 10px;border-radius:9px;font-weight:600;max-width:320px}
.modelpick span{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.modelpick:hover{background:var(--raise)}
.ctxwrap{margin-left:auto;display:flex;align-items:center;gap:10px;font-size:12.5px;color:var(--dim)}
.ctxbar{width:160px;height:6px;border-radius:6px;background:var(--raise);overflow:hidden}
.ctxbar i{display:block;height:100%;width:0;background:var(--grad);transition:width .4s}
.ctxbar.warn i{background:var(--warn)}.ctxbar.bad i{background:var(--bad)}
.pill{display:flex;align-items:center;gap:6px;padding:6px 11px;border:1px solid var(--line);border-radius:999px;font-size:12.5px;color:var(--dim)}
.pill:hover:not(:disabled){color:var(--text);background:var(--raise)}.pill:disabled{opacity:.4;cursor:default}
#log{flex:1;overflow:auto;scroll-behavior:smooth}
.col{max-width:780px;margin:0 auto;padding:22px 20px 40px}
.empty{height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:18px;padding:20px;text-align:center}
.empty h2{font-size:28px;font-weight:600;margin:0;background:var(--grad);-webkit-background-clip:text;background-clip:text;color:transparent}
.chips{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;max-width:640px}
.chip{border:1px solid var(--line);border-radius:12px;padding:10px 14px;color:var(--dim);font-size:13.5px;background:var(--panel)}.chip:hover{color:var(--text);background:var(--raise)}
.msg{margin:0 0 26px;position:relative}
.msg.user{display:flex;flex-direction:column;align-items:flex-end}
.bubble{background:var(--user);padding:10px 15px;border-radius:18px;max-width:85%;white-space:pre-wrap;word-wrap:break-word}
.msg.ai{display:flex;gap:14px}
.avatar{width:28px;height:28px;border-radius:50%;background:var(--grad);display:grid;place-items:center;flex-shrink:0;margin-top:2px}
.avatar svg{width:15px;height:15px;fill:#fff}
.body{flex:1;min-width:0;word-wrap:break-word}
.body p{margin:0 0 12px}.body ul,.body ol{margin:0 0 12px;padding-left:24px}.body li{margin:3px 0}
.body h1,.body h2,.body h3{margin:18px 0 8px;line-height:1.3}.body h1{font-size:22px}.body h2{font-size:19px}.body h3{font-size:16.5px}
.body a{color:var(--a2)}.body table{border-collapse:collapse;margin:0 0 12px;font-size:14px}.body td,.body th{border:1px solid var(--line);padding:5px 10px}
.body blockquote{margin:0 0 12px;padding-left:12px;border-left:3px solid var(--line);color:var(--dim)}
code{font-family:"Cascadia Code",Consolas,monospace;font-size:13.5px}
:not(pre)>code{background:var(--raise);padding:1px 6px;border-radius:5px}
.cb{border:1px solid var(--line);border-radius:12px;overflow:hidden;margin:0 0 14px;background:var(--code)}
.cbh{display:flex;align-items:center;gap:4px;padding:6px 8px 6px 14px;font-size:12px;color:var(--dim);background:var(--raise)}
.cbh b{font-weight:500;margin-right:auto}.cbh button{display:flex;align-items:center;gap:5px;padding:3px 8px;border-radius:6px;font-size:12px}
.cbh button:hover{background:var(--line);color:var(--text)}.cbh .icon{width:14px;height:14px}
.cb pre{margin:0;padding:14px;overflow:auto}.runout{border-top:1px solid var(--line);padding:10px 14px;font:12.5px Consolas,monospace;white-space:pre-wrap;color:var(--dim);max-height:300px;overflow:auto}
.runout iframe{width:100%;height:320px;border:0;background:#fff;border-radius:6px}
.tk-k{color:#c792ea}.tk-s{color:#a5d6a7}.tk-c{color:#7f848e;font-style:italic}.tk-n{color:#f5a97f}.tk-f{color:#82aaff}
[data-theme=light] .tk-k{color:#8b3fd1}[data-theme=light] .tk-s{color:#2e7d32}[data-theme=light] .tk-n{color:#c2410c}[data-theme=light] .tk-f{color:#1d4ed8}
details.think{border:1px solid var(--line);border-radius:12px;margin:0 0 14px;background:var(--panel)}
details.think summary{cursor:pointer;padding:8px 12px;color:var(--dim);font-size:13.5px;display:flex;align-items:center;gap:8px;list-style:none}
details.think summary::-webkit-details-marker{display:none}
details.think .tb{padding:2px 14px 10px;color:var(--dim);font-size:14px;border-top:1px solid var(--line)}
.pass{margin:10px 0}.pass h4{margin:0 0 4px;font-size:12.5px;color:var(--text);font-weight:600}
.actions{display:flex;gap:2px;margin-top:4px;opacity:0;transition:opacity .15s;align-items:center}
.msg:hover .actions,.msg.last .actions{opacity:1}
.actions button{padding:5px;border-radius:7px;color:var(--faint)}.actions button:hover{background:var(--raise);color:var(--text)}
.actions .icon{width:16px;height:16px}.meta{font-size:11.5px;color:var(--faint);margin-left:8px}
.memchip{display:inline-flex;align-items:center;gap:6px;font-size:12px;color:var(--dim);border:1px solid var(--line);border-radius:999px;padding:3px 10px;margin:0 6px 8px 0;background:var(--panel)}
.memchip .icon{width:13px;height:13px;color:var(--a1)}
.editbox{width:100%;max-width:85%;background:var(--user);border:1px solid var(--a1);border-radius:14px;padding:10px 14px;resize:vertical;min-height:70px}
.editbtns{display:flex;gap:8px;margin-top:8px}.btn{padding:7px 14px;border-radius:9px;border:1px solid var(--line);font-size:13.5px}
.btn:hover{background:var(--raise)}.btn.pri{background:var(--text);color:var(--bg);border-color:transparent;font-weight:600}.btn.pri:hover{opacity:.9}
.err{color:var(--bad);font-size:13.5px}
/* thinking hard */
.hard{display:flex;align-items:center;gap:12px;padding:12px 14px;border:1px solid var(--line);border-radius:14px;margin-bottom:14px;background:var(--panel);position:relative;overflow:hidden}
.hard::before{content:"";position:absolute;inset:0;background:linear-gradient(90deg,transparent,rgba(139,108,255,.12),transparent);animation:sweep 1.8s infinite}
@keyframes sweep{from{transform:translateX(-100%)}to{transform:translateX(100%)}}
.orbit{width:26px;height:26px;position:relative;flex-shrink:0}
.orbit i{position:absolute;inset:0;border-radius:50%;border:2px solid transparent;border-top-color:var(--a1);animation:spin 1s linear infinite}
.orbit i:nth-child(2){inset:4px;border-top-color:var(--a2);animation-duration:1.4s;animation-direction:reverse}
.orbit i:nth-child(3){inset:9px;background:var(--grad);border:0;animation:pulse 1.2s ease-in-out infinite}
@keyframes spin{to{transform:rotate(360deg)}}@keyframes pulse{50%{transform:scale(.55);opacity:.6}}
.hardtxt b{display:block;font-size:14px;background:var(--grad);-webkit-background-clip:text;background-clip:text;color:transparent}
.hardtxt span{font-size:12.5px;color:var(--dim)}
.steps{display:flex;gap:4px;margin-left:auto}.steps i{width:22px;height:4px;border-radius:4px;background:var(--line)}
.steps i.done{background:var(--grad)}.steps i.cur{background:var(--a1);animation:pulse 1s infinite}
.dots span{display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--dim);margin-right:4px;animation:pulse 1s infinite}
.dots span:nth-child(2){animation-delay:.15s}.dots span:nth-child(3){animation-delay:.3s}
/* composer */
.composer{padding:0 20px 14px}
.box{max-width:780px;margin:0 auto;border:1px solid var(--line);border-radius:22px;background:var(--panel);padding:10px 10px 8px 16px;box-shadow:var(--shadow)}
.box:focus-within{border-color:#4a4a55}
.box textarea{width:100%;border:0;background:transparent;resize:none;outline:0;max-height:240px;min-height:26px;padding:4px 0}
.boxrow{display:flex;align-items:center;gap:8px;margin-top:4px}
.toggle{display:flex;align-items:center;gap:6px;padding:6px 11px;border-radius:999px;border:1px solid var(--line);font-size:13px;color:var(--dim)}
.toggle:hover{color:var(--text)}.toggle.on{border-color:transparent;background:var(--grad);color:#fff}
.send{margin-left:auto;width:36px;height:36px;border-radius:50%;background:var(--text);color:var(--bg);display:grid;place-items:center}
.send:disabled{opacity:.3;cursor:default}.send .icon{stroke-width:2.4}
.under{max-width:780px;margin:8px auto 0;display:flex;align-items:center;gap:10px;font-size:12px;color:var(--faint)}
.ring{display:flex;align-items:center;gap:7px;cursor:pointer;position:relative}
.ring svg{width:22px;height:22px;transform:rotate(-90deg)}
.ringtip{position:absolute;bottom:30px;left:0;background:var(--raise);border:1px solid var(--line);border-radius:12px;padding:12px;width:300px;color:var(--dim);display:none;box-shadow:var(--shadow);z-index:5;font-size:12.5px;line-height:1.5}
.ring:hover .ringtip{display:block}.ringtip .bars{display:flex;flex-wrap:wrap;gap:2px;margin:8px 0}.ringtip .bars i{width:8px;height:14px;border-radius:2px}
.under .note{margin-left:auto}
/* live render card */
.rcard{max-width:420px;margin:10px 0;border:1px solid var(--line);border-radius:18px;overflow:hidden;background:var(--panel);box-shadow:var(--shadow)}
.rframe{position:relative;aspect-ratio:1;background:linear-gradient(135deg,#1d1f2b,#2a2140);overflow:hidden}
.rshimmer{position:absolute;inset:0;background:linear-gradient(110deg,transparent 30%,rgba(255,255,255,.08) 50%,transparent 70%);background-size:250% 100%;background-repeat:no-repeat;animation:shim 1.6s linear infinite}
@keyframes shim{from{background-position:150% 0}to{background-position:-100% 0}}
.rimg{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;opacity:0;filter:blur(14px) saturate(1.2);transform:scale(1.06);transition:opacity .5s,filter 1.2s,transform 1.2s;image-rendering:auto}
.rframe.has .rimg+.rimg{filter:blur(2px) saturate(1.1);transform:scale(1.01);transition:opacity .45s}   /* later previews only cross-fade */
.rframe.has .rimg.in{opacity:1;filter:blur(2px) saturate(1.1);transform:scale(1.01)}
.rframe.has .rshimmer{opacity:.35}
.rbadge{position:absolute;left:10px;bottom:10px;display:flex;align-items:center;gap:7px;padding:5px 11px;border-radius:999px;background:rgba(0,0,0,.55);backdrop-filter:blur(6px);color:#fff;font-size:12px}
.rdot{width:7px;height:7px;border-radius:50%;background:var(--a1);box-shadow:0 0 0 0 var(--a1);animation:pulse 1.2s infinite}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(140,120,255,.6)}100%{box-shadow:0 0 0 8px rgba(140,120,255,0)}}
.rfoot{display:flex;align-items:center;gap:10px;padding:10px 12px}
.rbar{flex:1;height:6px;border-radius:6px;background:var(--line);overflow:hidden}.rbar i{display:block;height:100%;width:3%;background:linear-gradient(90deg,var(--a1),var(--a2,var(--a1)));transition:width .4s}
.rpct{font-size:12px;color:var(--dim);min-width:34px;text-align:right}
.rfoot .btn{padding:5px 12px;font-size:12.5px}
.genimg{animation:reveal .6s ease}
@keyframes reveal{from{opacity:0;filter:blur(10px)}to{opacity:1;filter:none}}
.gwrap{position:relative;display:inline-block;margin:10px 0;max-width:100%}
.gwrap .genimg{margin:0;cursor:zoom-in;display:block}
.gbtns{position:absolute;top:10px;right:10px;display:flex;gap:6px;opacity:0;transition:opacity .2s}
.gwrap:hover .gbtns{opacity:1}
.gbtns button,.gbtns a{width:34px;height:34px;border-radius:10px;display:grid;place-items:center;background:rgba(0,0,0,.6);backdrop-filter:blur(6px);color:#fff}
.gbtns .icon{width:17px;height:17px}.gbtns button:hover,.gbtns a:hover{background:rgba(0,0,0,.8)}
.lbox{position:fixed;inset:0;z-index:50;background:rgba(5,5,10,.88);backdrop-filter:blur(8px);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:14px;padding:24px;animation:reveal .25s}
.lbox img{max-width:96vw;max-height:84vh;border-radius:14px;box-shadow:0 20px 60px rgba(0,0,0,.6);cursor:zoom-out}
.lbox .lbar{display:flex;gap:10px}
.lbox .lbar a,.lbox .lbar button{display:flex;align-items:center;gap:8px;padding:9px 16px;border-radius:999px;background:rgba(255,255,255,.12);color:#fff;font-size:14px}
.lbox .lbar a:hover,.lbox .lbar button:hover{background:rgba(255,255,255,.22)}
/* attachments */
.att{display:none;flex-wrap:wrap;gap:8px;margin:2px 0 8px}
.thumb{position:relative;width:64px;height:64px;border-radius:12px;overflow:hidden;border:1px solid var(--line)}
.thumb img{width:100%;height:100%;object-fit:cover}
.thumb button,.fchip button,.kf button{position:absolute;top:3px;right:3px;width:20px;height:20px;border-radius:50%;background:rgba(0,0,0,.65);color:#fff;font-size:13px;line-height:20px;text-align:center}
.fchip{position:relative;display:flex;align-items:center;gap:8px;padding:8px 32px 8px 10px;border:1px solid var(--line);border-radius:12px;font-size:13px;max-width:280px;background:var(--raise)}
.fchip span{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.fchip small{color:var(--faint);white-space:nowrap}
.fchip button{top:50%;transform:translateY(-50%);right:6px}
.kb{max-width:780px;margin:0 auto 8px;display:none;flex-wrap:wrap;align-items:center;gap:6px;font-size:12px;color:var(--dim)}
.kb>span:first-child{display:flex;align-items:center;gap:5px}.kb .icon{width:14px;height:14px}
.kf{position:relative;padding:3px 24px 3px 9px;border:1px solid var(--line);border-radius:999px;background:var(--raise)}
.kf button{top:50%;transform:translateY(-50%);right:3px;width:16px;height:16px;line-height:16px;font-size:11px}
.toggle.icononly{padding:6px 8px}
.box.drop{outline:2px dashed var(--a1);outline-offset:3px}
.uimgs{display:flex;flex-wrap:wrap;gap:6px;justify-content:flex-end;margin-bottom:6px;max-width:85%}
.uimgs img{max-width:220px;max-height:220px;border-radius:14px;border:1px solid var(--line);cursor:zoom-in}
.udocs{display:flex;flex-wrap:wrap;gap:6px;justify-content:flex-end;margin-bottom:6px}
.udocs .fchip{padding-right:10px}
/* model hub + system */
.seg{display:inline-flex;border:1px solid var(--line);border-radius:10px;padding:3px;gap:3px}.seg button{padding:6px 12px;border-radius:7px;color:var(--dim);font-size:13px}.seg button.on{background:var(--raise);color:var(--text)}
.hubrow{display:flex;gap:14px;align-items:flex-start;padding:12px 0;border-bottom:1px solid var(--line)}.hubmain{flex:1;min-width:0}.hubmain b{display:block;font-size:14px;word-break:break-word}
.hubmeta{display:block;font-size:12.5px;color:var(--dim);margin-top:3px}.hubmain p{margin:6px 0 0;font-size:12.5px;color:var(--faint);line-height:1.45}
.fit{display:inline-block;font-size:11.5px;padding:2px 8px;border-radius:20px;margin-left:4px;white-space:nowrap}.fit.fast{background:rgba(60,190,120,.15);color:#4cc38a}.fit.ok{background:rgba(230,180,60,.15);color:#e0b34a}.fit.slow{background:rgba(230,90,90,.15);color:#e57373}
.dlrow{border:1px solid var(--line);border-radius:12px;padding:10px 12px;margin:10px 0;font-size:13px}.dlrow b{display:block;word-break:break-all}.dlrow span{color:var(--dim);font-size:12px}
.dlbar{height:6px;border-radius:6px;background:var(--line);overflow:hidden;margin-top:8px}.dlbar i{display:block;height:100%;background:linear-gradient(90deg,var(--a1),var(--a2,var(--a1)));transition:width .4s}
.sysgrid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.syscard{border:1px solid var(--line);border-radius:12px;padding:12px 14px;font-size:13px}
.syscard h4{margin:0 0 8px;display:flex;align-items:center;gap:8px;font-size:13.5px}.syscard div{display:flex;justify-content:space-between;gap:10px;padding:3px 0}.syscard div span{color:var(--dim);flex-shrink:0}.syscard div b{font-weight:500;text-align:right;word-break:break-word}
.syscap{border:1px solid var(--line);border-radius:12px;padding:12px 14px;margin-top:12px;font-size:13px}.syscap h4{margin:0 0 6px}.syscap div{display:flex;gap:12px;align-items:baseline;padding:3px 0}.syscap b{width:64px;font-size:16px}.syscap span{color:var(--dim)}
.benchrow{display:grid;grid-template-columns:150px 1fr 110px;gap:12px;align-items:center;font-size:13px;padding:5px 0}.benchrow .dlbar{margin:0}.benchrow b{text-align:right;font-weight:600}
.sidefoot{flex-direction:column;align-items:stretch;gap:2px}
@media(max-width:820px){.sysgrid{grid-template-columns:1fr}.benchrow{grid-template-columns:100px 1fr 90px}}
/* settings */
.modal{position:fixed;inset:0;background:rgba(0,0,0,.55);display:none;align-items:center;justify-content:center;z-index:20;padding:20px}
.modal.open{display:flex}
.sheet{width:880px;max-width:100%;height:620px;max-height:100%;background:var(--panel);border:1px solid var(--line);border-radius:18px;display:flex;overflow:hidden;box-shadow:var(--shadow)}
.tabs{width:200px;border-right:1px solid var(--line);padding:16px 10px;display:flex;flex-direction:column;gap:2px;background:var(--side)}
.tabs h3{margin:0 8px 12px;font-size:16px}
.tabs button{display:flex;align-items:center;gap:10px;padding:8px 10px;border-radius:9px;color:var(--dim);text-align:left}
.tabs button.on,.tabs button:hover{background:var(--raise);color:var(--text)}
.pane{flex:1;overflow:auto;padding:22px 26px;position:relative}
.pane h2{margin:0 0 4px;font-size:19px}.pane .sub{color:var(--dim);font-size:13.5px;margin:0 0 18px}
.close{position:absolute;top:14px;right:14px}
.field{margin:0 0 16px}.field label{display:block;font-size:13px;font-weight:600;margin-bottom:6px}
.field small{display:block;color:var(--faint);font-size:12px;margin-top:4px}
.field input,.field textarea,.field select{width:100%;padding:8px 11px;border-radius:9px;border:1px solid var(--line);background:var(--bg)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.switch{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:12px 0;border-bottom:1px solid var(--line)}
.switch div b{display:block;font-size:14px;font-weight:600}.switch div span{font-size:12.5px;color:var(--dim)}
.sw{width:40px;height:22px;border-radius:22px;background:var(--line);position:relative;flex-shrink:0;transition:.2s}
.sw::after{content:"";position:absolute;top:3px;left:3px;width:16px;height:16px;border-radius:50%;background:#fff;transition:.2s}
.sw.on{background:var(--a1)}.sw.on::after{left:21px}
.memrow{display:flex;gap:10px;align-items:flex-start;padding:10px 12px;border:1px solid var(--line);border-radius:10px;margin-bottom:8px}
.memrow span{flex:1;font-size:14px}.memrow small{color:var(--faint);font-size:11px;display:block}
.memrow textarea{flex:1;border:1px solid var(--a1);border-radius:8px;background:var(--bg);padding:6px 8px}
.cmp{width:100%;border-collapse:collapse;font-size:13px}.cmp th,.cmp td{border-bottom:1px solid var(--line);padding:8px 6px;text-align:left;vertical-align:top}
.cmp th{color:var(--dim);font-weight:600}.cmp td:first-child{color:var(--dim);width:22%}
.pc{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:16px}.pc div{border:1px solid var(--line);border-radius:12px;padding:12px 14px;font-size:13px}
.pc h4{margin:0 0 6px}.pc ul{margin:0;padding-left:18px}.pc li{margin:3px 0}
.toast{position:fixed;bottom:22px;left:50%;transform:translateX(-50%);background:var(--raise);border:1px solid var(--line);padding:9px 16px;border-radius:10px;font-size:13.5px;box-shadow:var(--shadow);display:none;z-index:30}
.nav{display:flex;gap:4px;margin:0 12px 8px;padding:3px;border:1px solid var(--line);border-radius:11px;background:var(--panel)}
.nav button{flex:1;display:flex;flex-direction:column;align-items:center;gap:2px;padding:6px 2px;border-radius:8px;font-size:11.5px;color:var(--dim)}
.nav button .icon{width:16px;height:16px}.nav button.on{background:var(--raise);color:var(--text)}.nav button:hover{color:var(--text)}
#studio{flex:1;overflow:auto;display:none}#studio .col{max-width:860px}
.studio h1{font-size:24px;margin:6px 0 4px;display:flex;align-items:center;gap:10px}.studio h1 .icon{width:24px;height:24px;color:var(--a1)}
.banner{border:1px solid var(--line);border-left:3px solid var(--warn);border-radius:10px;padding:10px 14px;font-size:13.5px;color:var(--dim);margin:14px 0 18px;background:var(--panel)}
.card{border:1px solid var(--line);border-radius:14px;padding:16px 18px;background:var(--panel);margin-bottom:16px}
.card h3{margin:0 0 12px;font-size:15px}.opts{display:flex;flex-wrap:wrap;gap:6px}
.opt{padding:6px 12px;border:1px solid var(--line);border-radius:999px;font-size:13px;color:var(--dim)}.opt.on{border-color:transparent;background:var(--grad);color:#fff}
.preview{background:var(--code);border:1px dashed var(--line);border-radius:10px;padding:12px;font-size:13.5px;white-space:pre-wrap;min-height:60px}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px}.gallery .ph{aspect-ratio:1;border:1px dashed var(--line);border-radius:12px;display:grid;place-items:center;color:var(--faint);font-size:12px}
.genimg{max-width:100%;border-radius:12px;margin:6px 0}
.result{margin-top:12px;font-size:13.5px}
@media(max-width:820px){aside{position:fixed;z-index:10;height:100%}.ctxbar{width:80px}.sheet{flex-direction:column;height:90vh}.tabs{width:auto;flex-direction:row;overflow:auto}.tabs h3{display:none}.grid2,.pc{grid-template-columns:1fr}}
/* every form control follows the theme (some had the browser's white default, hiding light text) */
:root{color-scheme:dark}:root[data-theme=light]{color-scheme:light}
input:not([type=checkbox]):not([type=radio]):not([type=range]):not([type=file]),textarea:not(#q),select{background:var(--bg);color:var(--text);border:1px solid var(--line);border-radius:9px;font:inherit}
input::placeholder,textarea::placeholder{color:var(--faint)}
select option{background:var(--panel);color:var(--text)}
input[type=checkbox],input[type=radio],input[type=range]{accent-color:var(--a1)}
input[type=file]{color:var(--dim)}
/* waiting label, loading shine */
.dots span{animation-delay:var(--ph,0ms)}.dots span:nth-child(2){animation-delay:calc(var(--ph,0ms) + .15s)}.dots span:nth-child(3){animation-delay:calc(var(--ph,0ms) + .3s)}
.waiting{display:inline-flex;align-items:center;gap:10px;min-height:22px}
.wl{font-size:13px;color:var(--dim)}
.wl:not(:empty){background:linear-gradient(90deg,var(--dim) 30%,var(--text) 50%,var(--dim) 70%);background-size:250% 100%;background-repeat:no-repeat;-webkit-background-clip:text;background-clip:text;color:transparent;animation:wshine 1.8s linear infinite}
@keyframes wshine{from{background-position:120% 0}to{background-position:-120% 0}}
/* new chat: the spark pops in and twinkles, the heading and suggestions rise in */
.empty .logo{position:relative;animation:spawn .7s cubic-bezier(.2,1.5,.4,1) both}
.empty .logo svg{animation:twinkle 3.2s ease-in-out .7s infinite}
.empty .logo::after{content:"";position:absolute;inset:-7px;border-radius:18px;border:1.5px solid rgba(139,108,255,.55);opacity:0;animation:halo 3.2s ease-out .7s infinite}
.empty .spark{position:absolute;width:6px;height:6px;border-radius:50%;background:#fff;box-shadow:0 0 8px #b9a8ff;opacity:0;animation:glint 3.2s ease-in-out infinite}
.empty h2{animation:rise .55s .12s cubic-bezier(.2,.9,.3,1) both}
.empty .chip{animation:rise .5s cubic-bezier(.2,.9,.3,1) both}
.empty .chip:nth-child(1){animation-delay:.22s}.empty .chip:nth-child(2){animation-delay:.29s}.empty .chip:nth-child(3){animation-delay:.36s}.empty .chip:nth-child(4){animation-delay:.43s}
.empty .chip{transition:transform .15s,border-color .15s}.empty .chip:hover{transform:translateY(-2px);border-color:var(--a1)}
@keyframes spawn{from{transform:scale(0) rotate(-90deg);opacity:0}to{transform:none;opacity:1}}
@keyframes twinkle{0%,100%{transform:scale(1) rotate(0)}12%{transform:scale(1.18) rotate(18deg)}24%{transform:scale(1) rotate(0)}}
@keyframes halo{0%{opacity:0;transform:scale(.8)}12%{opacity:.9}40%,100%{opacity:0;transform:scale(1.35)}}
@keyframes glint{0%,100%{opacity:0;transform:scale(.2)}8%{opacity:1;transform:scale(1)}18%{opacity:0;transform:scale(.2)}}
@keyframes rise{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}
/* sending from a new chat: suggestions pop away, the heading lifts off */
.empty.leaving .chip{animation:popout .3s cubic-bezier(.5,0,.75,0) forwards}
.empty.leaving .chip:nth-child(2){animation-delay:.04s}.empty.leaving .chip:nth-child(3){animation-delay:.08s}.empty.leaving .chip:nth-child(4){animation-delay:.12s}
.empty.leaving h2{animation:liftoff .35s cubic-bezier(.5,0,.75,0) forwards}
@keyframes popout{to{opacity:0;transform:scale(.6) translateY(10px)}}
@keyframes liftoff{to{opacity:0;transform:translateY(-18px) scale(.96);filter:blur(4px)}}
.flylogo{position:fixed;z-index:50;pointer-events:none;border-radius:14px;background:var(--grad);display:grid;place-items:center;box-shadow:0 0 22px rgba(139,108,255,.6)}
.flylogo svg{width:58%;height:58%}
.avatar.land{animation:land .45s cubic-bezier(.2,1.6,.4,1)}
@keyframes land{0%{transform:scale(1.35)}100%{transform:scale(1)}}
/* new messages slide in */
.msg.fresh{animation:rise .35s cubic-bezier(.2,.9,.3,1) both}
.col>.msg:first-child{margin-top:34px}
/* 1. streamed text: each new paragraph, list item or block eases in */
.chunk{animation:chunk .38s cubic-bezier(.2,.9,.3,1) both}
@keyframes chunk{from{opacity:0;transform:translateY(6px);filter:blur(3px)}to{opacity:1;transform:none;filter:none}}
/* 2. send <-> stop morph */
.send{transition:background .25s,color .25s,transform .15s}.send:active{transform:scale(.92)}
.send.morph .icon{animation:morph .38s cubic-bezier(.3,1.4,.5,1)}
.send.busy{background:var(--grad);color:#fff;box-shadow:0 0 0 0 rgba(139,108,255,.5);animation:busyring 1.8s ease-out infinite}
@keyframes morph{0%{transform:rotate(-120deg) scale(.3);opacity:0}100%{transform:none;opacity:1}}
@keyframes busyring{0%{box-shadow:0 0 0 0 rgba(139,108,255,.45)}70%,100%{box-shadow:0 0 0 9px rgba(139,108,255,0)}}
/* 3. a finished picture sharpens in and a light sweep passes over it once */
.gwrap.justmade{overflow:hidden;border-radius:12px}
.gwrap.justmade .genimg{animation:sharpen 1.1s cubic-bezier(.2,.8,.2,1) both}
.gwrap.justmade::after{content:"";position:absolute;inset:0;pointer-events:none;background:linear-gradient(105deg,transparent 35%,rgba(255,255,255,.35) 50%,transparent 65%);transform:translateX(-110%);animation:sweep1 1.2s .5s ease-out forwards}
@keyframes sharpen{from{filter:blur(16px) saturate(1.3);transform:scale(1.04)}to{filter:none;transform:none}}
@keyframes sweep1{to{transform:translateX(110%)}}
/* 4. new chats slide into the history; switching views cross-fades */
.hitem.slidein{animation:slidein .4s cubic-bezier(.2,.9,.3,1) both}
@keyframes slidein{from{opacity:0;transform:translateX(-14px);max-height:0}to{opacity:1;transform:none;max-height:40px}}
.viewin{animation:viewin .32s cubic-bezier(.2,.9,.3,1) both}
@keyframes viewin{from{opacity:0;transform:translateY(8px) scale(.995)}to{opacity:1;transform:none}}
/* studios */
.hero textarea,.sizerow input[type=number]{padding:8px 11px;border-radius:9px;border:1px solid var(--line);background:var(--bg);color:var(--text);font:inherit}
.hero textarea{width:100%;font-size:15px;min-height:84px;resize:vertical;padding:12px 14px;border-radius:12px;transition:border-color .2s,box-shadow .2s}
.hero textarea:focus,.sizerow input:focus{outline:none;border-color:var(--a1);box-shadow:0 0 0 3px rgba(139,108,255,.18)}
.sizerow input[type=range]{accent-color:var(--a1)}
.sizerow{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:12px 0}
.sizerow label{font-size:13px;font-weight:600;color:var(--dim)}
.sizerow input[type=number]{width:86px}
.sizerow .x{color:var(--faint)}
.sizerow .hint{font-size:12.5px;color:var(--faint)}
.chk{display:inline-flex;align-items:center;gap:6px;font-weight:500!important;cursor:pointer}
.studio .row{display:flex;gap:8px;align-items:center}
details.adv{padding:0}details.adv>summary{padding:14px 18px;cursor:pointer;font-weight:600;font-size:14px;list-style:none;display:flex;align-items:center;gap:8px}
details.adv>summary::before{content:"›";display:inline-block;transition:transform .2s;color:var(--dim)}details.adv[open]>summary::before{transform:rotate(90deg)}
details.adv>.in{padding:0 18px 16px}
.studio .card{animation:rise .4s cubic-bezier(.2,.9,.3,1) both}.studio .card:nth-of-type(2){animation-delay:.05s}.studio .card:nth-of-type(3){animation-delay:.1s}
.btn.pri{transition:transform .12s,box-shadow .2s}.btn.pri:hover{box-shadow:0 6px 20px rgba(139,108,255,.35)}.btn.pri:active{transform:scale(.97)}
</style></head><body>
<svg width="0" height="0" style="position:absolute"><defs>
<symbol id="i-plus" viewBox="0 0 24 24"><path d="M12 5v14M5 12h14"/></symbol>
<symbol id="i-gear" viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/></symbol>
<symbol id="i-menu" viewBox="0 0 24 24"><path d="M3 6h18M3 12h18M3 18h18"/></symbol>
<symbol id="i-x" viewBox="0 0 24 24"><path d="M18 6 6 18M6 6l12 12"/></symbol>
<symbol id="i-trash" viewBox="0 0 24 24"><path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6"/></symbol>
<symbol id="i-copy" viewBox="0 0 24 24"><rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15H4a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v1"/></symbol>
<symbol id="i-redo" viewBox="0 0 24 24"><path d="M21 12a9 9 0 1 1-3-6.7L21 8"/><path d="M21 3v5h-5"/></symbol>
<symbol id="i-edit" viewBox="0 0 24 24"><path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/></symbol>
<symbol id="i-up" viewBox="0 0 24 24"><path d="M12 19V5M5 12l7-7 7 7"/></symbol>
<symbol id="i-stop" viewBox="0 0 24 24"><rect x="7" y="7" width="10" height="10" rx="1.5" fill="currentColor"/></symbol>
<symbol id="i-brain" viewBox="0 0 24 24"><path d="M9.5 2A2.5 2.5 0 0 0 7 4.5v.3A3 3 0 0 0 4.5 8a3 3 0 0 0-.8 5.2A3 3 0 0 0 6 18a3 3 0 0 0 3.5 3A2.5 2.5 0 0 0 12 19.5v-15A2.5 2.5 0 0 0 9.5 2zM14.5 2A2.5 2.5 0 0 1 17 4.5v.3A3 3 0 0 1 19.5 8a3 3 0 0 1 .8 5.2A3 3 0 0 1 18 18a3 3 0 0 1-3.5 3 2.5 2.5 0 0 1-2.5-1.5"/></symbol>
<symbol id="i-compact" viewBox="0 0 24 24"><path d="M4 14h6v6M20 10h-6V4M14 10l7-7M3 21l7-7"/></symbol>
<symbol id="i-mem" viewBox="0 0 24 24"><path d="M12 2a7 7 0 0 0-4 12.7V17h8v-2.3A7 7 0 0 0 12 2zM9 21h6"/></symbol>
<symbol id="i-chat" viewBox="0 0 24 24"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></symbol>
<symbol id="i-chip" viewBox="0 0 24 24"><rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/></symbol>
<symbol id="i-code" viewBox="0 0 24 24"><path d="m16 18 6-6-6-6M8 6l-6 6 6 6"/></symbol>
<symbol id="i-dl" viewBox="0 0 24 24"><path d="M12 3v12M7 10l5 5 5-5M4 21h16"/></symbol>
<symbol id="i-gauge" viewBox="0 0 24 24"><path d="M12 14l4-4M3.3 17a9 9 0 1 1 17.4 0"/><circle cx="12" cy="14" r="1.5"/></symbol>
<symbol id="i-clip" viewBox="0 0 24 24"><path d="m21.4 11.6-9.2 9.2a6 6 0 0 1-8.5-8.5l9.2-9.2a4 4 0 0 1 5.7 5.7l-9.2 9.2a2 2 0 0 1-2.8-2.8l8.5-8.5"/></symbol>
<symbol id="i-file" viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6M8 13h8M8 17h6"/></symbol>
<symbol id="i-expand" viewBox="0 0 24 24"><path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"/></symbol>
<symbol id="i-power" viewBox="0 0 24 24"><path d="M12 2v10M18.4 6.6a9 9 0 1 1-12.8 0"/></symbol>
<symbol id="i-info" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/></symbol>
<symbol id="i-play" viewBox="0 0 24 24"><path d="m6 4 14 8-14 8z"/></symbol>
<symbol id="i-eye" viewBox="0 0 24 24"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></symbol>
<symbol id="i-down" viewBox="0 0 24 24"><path d="m6 9 6 6 6-6"/></symbol>
<symbol id="i-check" viewBox="0 0 24 24"><path d="M20 6 9 17l-5-5"/></symbol>
<symbol id="i-user" viewBox="0 0 24 24"><circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/></symbol>
<symbol id="i-pin" viewBox="0 0 24 24"><path d="M12 17v5M9 3h6l-1 6 4 4H6l4-4z"/></symbol>
<symbol id="i-image" viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="m21 15-5-5L5 21"/></symbol>
<symbol id="i-video" viewBox="0 0 24 24"><rect x="2" y="6" width="14" height="12" rx="2"/><path d="m22 8-6 4 6 4z"/></symbol>
<symbol id="i-audio" viewBox="0 0 24 24"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/></symbol>
<symbol id="i-wand" viewBox="0 0 24 24"><path d="m15 4 5 5L8 21H3v-5zM13 6l5 5M19 2v3M22 4h-3"/></symbol>
<symbol id="spark" viewBox="0 0 24 24"><path d="M12 1.5l2.3 6.6 6.7 2.4-6.7 2.4L12 19.5l-2.3-6.6L3 10.5l6.7-2.4z"/><path d="M19 15l1 2.6 2.5 1-2.5 1L19 22l-1-2.4-2.5-1 2.5-1z"/></symbol>
</defs></svg>

<aside id="side">
  <div class="brand"><div class="logo"><svg><use href="#spark"/></svg></div>Ithaca</div>
  <div class="nav" id="nav"><button data-view="chat" class="on"><svg class="icon"><use href="#i-chat"/></svg>Chat</button><button data-view="image"><svg class="icon"><use href="#i-image"/></svg>Image</button><button data-view="video"><svg class="icon"><use href="#i-video"/></svg>Video</button><button data-view="audio"><svg class="icon"><use href="#i-audio"/></svg>Audio</button></div>
  <button class="newchat" id="newchat"><svg class="icon"><use href="#i-plus"/></svg>New chat</button>
  <input class="search" id="search" placeholder="Search chats">
  <div id="history"></div>
  <div class="sidefoot"><button id="openHub"><svg class="icon"><use href="#i-dl"/></svg>Models</button><button id="openSettings"><svg class="icon"><use href="#i-gear"/></svg>Settings</button><button id="quitApp" title="Stop Ithaca and free all its memory"><svg class="icon"><use href="#i-power"/></svg>Quit</button></div>
</aside>

<main>
  <div class="topbar">
    <button class="tbtn" id="toggleSide" title="Toggle sidebar"><svg class="icon"><use href="#i-menu"/></svg></button>
    <button class="modelpick" id="modelPick" title="Choose model"><span id="modelName">No model loaded</span><svg class="icon"><use href="#i-down"/></svg></button>
    <div class="ctxwrap">
      <span id="ctxText" title="Tokens this chat uses out of the model's context window">0 / 0</span>
      <div class="ctxbar" id="ctxBar"><i></i></div>
      <button class="pill" id="compact" title="Summarize older messages to free context space"><svg class="icon" style="width:14px;height:14px"><use href="#i-compact"/></svg>Compact</button>
    </div>
  </div>
  <div id="log"></div>
  <div id="studio"></div>
  <div class="composer">
    <div class="kb" id="kb"></div>
    <div class="box" id="box">
      <div class="att" id="att"></div>
      <textarea id="q" rows="1" placeholder="Message Ithaca"></textarea>
      <div class="boxrow">
        <button class="toggle icononly" id="attBtn" title="Attach images or files (PDF, Word, PowerPoint, Excel, text, code). You can also paste or drop them."><svg class="icon" style="width:16px;height:16px"><use href="#i-clip"/></svg></button>
        <input type="file" id="fileIn" multiple hidden accept="image/*,.pdf,.docx,.pptx,.xlsx,.txt,.md,.csv,.tsv,.json,.xml,.html,.htm,.log,.ini,.yaml,.yml,.toml,.py,.js,.ts,.tsx,.jsx,.c,.cpp,.h,.hpp,.cs,.java,.go,.rs,.rb,.php,.lua,.gd,.sh,.bat,.ps1,.sql,.tex,.rtf">
        <button class="toggle" id="hardBtn" title="Think hard: the model makes a short plan (and checks it when the task is tricky) before answering"><svg class="icon" style="width:15px;height:15px"><use href="#i-brain"/></svg>Think hard</button>
        <button class="send" id="send" disabled title="Send"><svg class="icon"><use href="#i-up"/></svg></button>
      </div>
    </div>
    <div class="under">
      <div class="ring" id="ring"><svg viewBox="0 0 36 36" id="ringSvg"></svg><span id="ringText">no model</span>
        <div class="ringtip" id="ringTip"></div></div>
      <span class="note">Runs fully on this PC. Can make mistakes.</span>
    </div>
  </div>
</main>

<div class="modal" id="settings"><div class="sheet">
  <div class="tabs" id="tabs"><h3>Settings</h3>
    <button data-tab="general" class="on"><svg class="icon"><use href="#i-user"/></svg>General</button>
    <button data-tab="model"><svg class="icon"><use href="#i-chip"/></svg>Model &amp; hardware</button>
    <button data-tab="hub"><svg class="icon"><use href="#i-dl"/></svg>Download models</button>
    <button data-tab="system"><svg class="icon"><use href="#i-gauge"/></svg>System &amp; benchmark</button>
    <button data-tab="memory"><svg class="icon"><use href="#i-mem"/></svg>Memory</button>
    <button data-tab="code"><svg class="icon"><use href="#i-code"/></svg>Coding</button>
    <button data-tab="about"><svg class="icon"><use href="#i-info"/></svg>About &amp; compare</button>
  </div>
  <div class="pane" id="pane"></div>
  <button class="tbtn close" id="closeSettings"><svg class="icon"><use href="#i-x"/></svg></button>
</div></div>
<div class="toast" id="toast"></div>

<script>
const $=id=>document.getElementById(id);
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const ico=n=>`<svg class="icon"><use href="#i-${n}"/></svg>`;
const uid=()=>Date.now().toString(36)+Math.random().toString(36).slice(2,8);
async function api(u,b){const r=await fetch(u,b===undefined?{}:{method:'POST',body:JSON.stringify(b)});return r.json()}
function toast(t){const e=$('toast');e.textContent=t;e.style.display='block';clearTimeout(toast.t);toast.t=setTimeout(()=>e.style.display='none',2600)}

let prefs={name:'',about:'',style:'',instructions:'',temperature:0.7,max_tokens:2048,memory:true,run_code:true,theme:'dark',spec_decode:true};
let memories=[],chats=[],chat=null,busy=false,hard=false,status={},abortCtl=null;

/* ---------- markdown ---------- */
const KW='abstract and as assert async auto await bool boolean break case catch char class const constexpr continue def default del delete do double elif else enum except export extends false final finally float fn for from func function global go if impl import in include int interface is lambda let long match mut namespace new nil none not null or package pass private protected pub public raise return self short signed static string struct super switch template this throw throws true try type typedef typeof uint union unsigned use using var void volatile while with yield None True False'.split(' ');
const KWS=new Set(KW);
function hl(code){
  const re=/(\/\/[^\n]*|#[^\n]*|\/\*[\s\S]*?\*\/|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'|`(?:\\.|[^`\\])*`|\b\d+(?:\.\d+)?\b|\b[A-Za-z_]\w*(?=\s*\()|\b[A-Za-z_]\w*\b)/g;
  let out='',last=0,m;
  while((m=re.exec(code))){out+=esc(code.slice(last,m.index));const t=m[0];let c='';
    if(/^(\/\/|#|\/\*)/.test(t))c='c';else if(/^["'`]/.test(t))c='s';else if(/^\d/.test(t))c='n';else if(KWS.has(t))c='k';else if(/^[A-Za-z_]/.test(t)&&code.slice(m.index+t.length).match(/^\s*\(/))c='f';
    out+=c?`<span class="tk-${c}">${esc(t)}</span>`:esc(t);last=m.index+t.length}
  return out+esc(code.slice(last))}
function inline(s){s=s.replace(/!\[[^\]]*\]\((?!https?:|\/outputs\/)[^)]*\)/g,'');return esc(s).replace(/`([^`]+)`/g,'<code>$1</code>').replace(/\*\*([^*]+)\*\*/g,'<b>$1</b>').replace(/(^|[^*])\*([^*\s][^*]*)\*/g,'$1<i>$2</i>').replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g,'<a href="$2" target="_blank" rel="noopener">$1</a>')}
function md(src){
  const parts=src.split(/```/);let html='';
  parts.forEach((p,i)=>{
    if(i%2){const nl=p.indexOf('\n');const lang=(nl>=0?p.slice(0,nl):'').trim();const code=nl>=0?p.slice(nl+1).replace(/\n$/,''):p;
      const id='cb'+uid();const runnable=/^(py|python)$/i.test(lang)&&prefs.run_code;const previewable=/^(html|svg)$/i.test(lang);
      html+=`<div class="cb" id="${id}"><div class="cbh"><b>${esc(lang||'code')}</b>`+
        (runnable?`<button onclick="runCode('${id}')">${ico('play')}Run</button>`:'')+(previewable?`<button onclick="preview('${id}')">${ico('eye')}Preview</button>`:'')+
        `<button onclick="copyCode('${id}')">${ico('copy')}Copy</button></div><pre><code>${hl(code)}</code></pre></div>`;return}
    const lines=p.split('\n');let para=[],list=null;
    const flush=()=>{if(para.length){html+='<p>'+para.map(inline).join('<br>')+'</p>';para=[]}if(list){html+=`<${list.t}>`+list.items.map(x=>'<li>'+inline(x)+'</li>').join('')+`</${list.t}>`;list=null}};
    for(let k=0;k<lines.length;k++){const l=lines[k];let m;
      if(!l.trim()){flush();continue}
      if((m=l.match(/^(#{1,3})\s+(.*)/))){flush();html+=`<h${m[1].length}>${inline(m[2])}</h${m[1].length}>`;continue}
      if(/^\|.*\|\s*$/.test(l)&&lines[k+1]&&/^\|[\s:|-]+\|\s*$/.test(lines[k+1])){flush();const row=x=>x.trim().replace(/^\||\|$/g,'').split('|');
        html+='<table><tr>'+row(l).map(c=>'<th>'+inline(c.trim())+'</th>').join('')+'</tr>';k+=2;
        while(k<lines.length&&/^\|/.test(lines[k])){html+='<tr>'+row(lines[k]).map(c=>'<td>'+inline(c.trim())+'</td>').join('')+'</tr>';k++}k--;html+='</table>';continue}
      if((m=l.match(/^\s*[-*+]\s+(.*)/))){if(para.length)flush();if(!list||list.t!='ul'){flush();list={t:'ul',items:[]}}list.items.push(m[1]);continue}
      if((m=l.match(/^\s*\d+[.)]\s+(.*)/))){if(para.length)flush();if(!list||list.t!='ol'){flush();list={t:'ol',items:[]}}list.items.push(m[1]);continue}
      if((m=l.match(/^>\s?(.*)/))){flush();html+='<blockquote>'+inline(m[1])+'</blockquote>';continue}
      if(list){flush()}para.push(l)}
    flush()});
  return html}
window.copyCode=id=>{navigator.clipboard.writeText($(id).querySelector('code').innerText);toast('Copied')};
window.runCode=async id=>{const cb=$(id);let o=cb.querySelector('.runout');if(!o){o=document.createElement('div');o.className='runout';cb.appendChild(o)}
  o.innerHTML='<span class="dots"><span></span><span></span><span></span></span> running…';
  const r=await api('/api/run',{code:cb.querySelector('code').innerText});o.textContent=r.output||'(no output)'};
window.preview=id=>{const cb=$(id);let o=cb.querySelector('.runout');if(!o){o=document.createElement('div');o.className='runout';cb.appendChild(o)}
  o.innerHTML='';const f=document.createElement('iframe');f.sandbox='allow-scripts';f.srcdoc=cb.querySelector('code').innerText;o.appendChild(f)};

/* ---------- memory + tools ---------- */
const CATS={about:'About you',preferences:'Preferences',projects:'Projects',knowledge:'Knowledge',other:'Other'};
// Tool tags the model may write. An unclosed tag at the very end (reply cut off) still counts.
const TAG_RE=/<(memory_save|memory_edit|memory_delete|memory_search|generate_image|generate_video|generate_audio)\b([^>]*?)(?:\/>|>([\s\S]*?)(?:<\/\1>|$))/g;
function attrs(s){const o={};(s||'').replace(/(\w+)\s*=\s*"([^"]*)"/g,(_,k,v)=>{o[k]=v});(s||'').replace(/(\w+)\s*=\s*([\w.-]+)/g,(_,k,v)=>{if(!(k in o))o[k]=v});return o}
function splitThink(t){const m=t.match(/^\s*(?:<think>)?([\s\S]*?)<\/think>([\s\S]*)$/);
  if(m)return{think:m[1].trim(),body:m[2]};if(/^\s*<think>/.test(t))return{think:t.replace(/^\s*<think>/,'').trim(),body:'',open:true};return{think:'',body:t}}
// Tools are only read from the final answer, never from the model's private reasoning.
function parseTools(text){const body=splitThink(text).body,out=[];let m;TAG_RE.lastIndex=0;
  while((m=TAG_RE.exec(body))){const a=attrs(m[2]);let val=(m[3]||'').replace(/<\/?[a-z_]+[^>]*>/g,'').trim();if(!val)val=(a.prompt||a.text||a.description||'').trim();out.push({tag:m[1],a,val})}
  const alt=/\[\s*(generate_(?:image|video|audio))\s*[:=]\s*([^\]]+)\]/g;while((m=alt.exec(body)))out.push({tag:m[1],a:{},val:m[2].trim()});
  return out}
function stripTags(t){return t.replace(TAG_RE,'').replace(/<\/?(memory_|generate_)[^>]*>/g,'').replace(/<(memory_|generate_)[\s\S]*$/,'')}
const memNextId=()=>Math.max(0,...memories.map(x=>x.id))+1;
const normCat=c=>{c=String(c||'').toLowerCase();return CATS[c]?c:'other'};
function applyMemoryOps(tools){const ops=[];
  for(const t of tools){
    if(t.tag=='memory_save'&&t.val&&t.val.length<600){
      if(memories.some(m=>m.text.toLowerCase()==t.val.toLowerCase()))continue;
      // only facts about the user ("User prefers ..."), never notes about the assistant itself
      if(!/\buser\b/i.test(t.val)&&!(prefs.name&&t.val.toLowerCase().includes(prefs.name.toLowerCase())))continue;
      memories.push({id:memNextId(),text:t.val,category:normCat(t.a.category),pinned:false,created:Date.now(),updated:Date.now()});ops.push({k:'Saved memory',t:t.val})}
    else if(t.tag=='memory_edit'){const x=memories.find(m=>m.id==+t.a.id);
      if(x&&(t.val||t.a.category)){if(t.val)x.text=t.val;if(t.a.category)x.category=normCat(t.a.category);x.updated=Date.now();ops.push({k:'Updated memory',t:x.text})}}
    else if(t.tag=='memory_delete'){const i=memories.findIndex(m=>m.id==+t.a.id);if(i>=0){ops.push({k:'Forgot',t:memories[i].text});memories.splice(i,1)}}}
  if(dedupe())ops.push({k:'Merged duplicate memories',t:''});
  if(ops.length)api('/api/memories',memories);
  return ops}
// Drop memories whose wording overlaps another by 75%+ (keeps the longer text under the older id).
function dedupe(){const words=t=>new Set(t.toLowerCase().match(/[a-z0-9+#]+/g)||[]);let removed=false;
  for(let i=0;i<memories.length;i++)for(let j=memories.length-1;j>i;j--){const a=words(memories[i].text),b=words(memories[j].text);
    const inter=[...a].filter(x=>b.has(x)).length,sim=inter/Math.max(1,Math.min(a.size,b.size));
    if(sim>=.75&&Math.min(a.size,b.size)>=3){if(memories[j].text.length>memories[i].text.length)memories[i].text=memories[j].text;memories[i].pinned=memories[i].pinned||memories[j].pinned;memories[i].updated=Date.now();memories.splice(j,1);removed=true}}
  return removed}
function searchMem(q,k=8){const w=new Set(String(q).toLowerCase().match(/[a-z0-9+#]{3,}/g)||[]);
  return memories.map(m=>{const t=(m.text+' '+m.category).toLowerCase();let sc=0;w.forEach(x=>{if(t.includes(x))sc++});return[sc,m]}).filter(x=>x[0]>0).sort((a,b)=>b[0]-a[0]).slice(0,k).map(x=>x[1])}
const memLine=m=>`[${m.id}] (${normCat(m.category)}) ${m.text}`;

/* ---------- prompt building ---------- */
function systemPrompt(extra,lastUser){
  let s=`You are Ithaca, a capable and honest AI assistant running entirely on the user's own PC, with no internet access. Today is ${new Date().toDateString()}.

# How to answer
- Lead with the direct answer, then add detail only where it helps. Match length to the question: short questions get short answers.
- Use Markdown: short paragraphs, bullet lists for steps or options, tables for comparisons, **bold** for key terms.
- Be accurate. If you are unsure or missing information, say so plainly instead of guessing, and ask a clarifying question when a request is ambiguous.
- Never invent tools, files, results or abilities you do not have.

# Coding
- Put code in fenced blocks with a language tag (\`\`\`python, \`\`\`cpp, ...).
- Give complete, runnable code: imports, an entry point, no "..." placeholders. Prefer the standard library and simple solutions.
- When fixing code, show the full corrected function or file, then explain what changed and why.
- After the code, explain briefly how it works and how to run it. The user can run Python blocks with one click.`;
  if(prefs.name||prefs.about||prefs.style||prefs.instructions){s+='\n\n# About the user';
    if(prefs.name)s+=`\n- Name: ${prefs.name}`;if(prefs.about)s+=`\n- ${prefs.about}`;
    if(prefs.style)s+=`\n- Preferred response style: ${prefs.style}`;if(prefs.instructions)s+=`\n- Instructions from the user: ${prefs.instructions}`}
  s+=`

# Tools
Call a tool by writing its tag in your final answer. Tags are invisible to the user; results come back to you automatically and you then continue the reply.
- <generate_image width="W" height="H">detailed visual description</generate_image>: make ONE image when the user asks for one. Set width and height to the shape they need (phone wallpaper width="1170" height="2532", desktop wallpaper width="1920" height="1080", portrait width="832" height="1216", otherwise width="1024" height="1024"). Ithaca renders the closest size the image model handles well. Write the tag once, with plain text inside and nothing else. Order: one or two sentences saying what you are making, then the tag, then one short line after it (the picture appears between them).
- <generate_video seconds="5">detailed description of subject, action, camera and style</generate_video>: make a short video.
- <generate_audio kind="speech">exact text to speak</generate_audio>: say something out loud in the user's chosen voice (when they ask you to speak, read aloud or make a voice clip).
If a tool reports a problem (for example that no model is loaded), explain it to the user in your own words and suggest what they can do.`;
  if(prefs.memory){
    let list=memories;
    if(memories.length>40){const rel=new Set(searchMem(lastUser||'',15).map(m=>m.id));list=memories.filter(m=>m.pinned||rel.has(m.id))}
    s+=`

# Long-term memory
You remember things about the user across all chats. ${memories.length?`Saved memories (${list.length} of ${memories.length} shown):\n`+list.map(memLine).join('\n'):'No memories are saved yet.'}
Memory tags (hidden from the user):
- <memory_save category="about|preferences|projects|knowledge|other">one short, self-contained fact</memory_save>
- <memory_edit id="N">the full corrected text</memory_edit> (you may also set category="...")
- <memory_delete id="N"/>
- <memory_search>keywords</memory_search>: look up memories not shown above.
Rules:
- Save lasting information: who the user is, their preferences, tools they use, ongoing projects, and anything they ask you to remember. Do not save small talk, one-off tasks or secrets (passwords, codes, keys, account numbers).
- Never save what is in attached images or files: they stay available in this chat. Only save it if the user explicitly asks you to remember it.
- Check the list first. If a related memory exists, edit it instead of saving a duplicate. Write in third person ("User prefers ...").
- When the user corrects something or asks you to forget it, edit or delete that memory.
- Write tags only in your final answer, never while reasoning, and never mention the tags themselves.
- Most messages need no memory tag at all. Questions, requests and small talk are not facts about the user.
- Only when you actually wrote a memory tag in this reply, say so naturally (e.g. "Got it, I'll remember that."). Otherwise never say you will remember something.`}
  if(chat&&chat.summary)s+='\n\n# Summary of the earlier part of this conversation\n'+chat.summary;
  const fc=fileContext(lastUser||'');if(fc)s+='\n\n'+fc;
  if(extra)s+='\n\n'+extra;
  return s}
const REMINDER='\n\n(Reminder from Ithaca, not from the user: you can create media yourself. Image: <generate_image>description</generate_image>. Video: <generate_video>description</generate_video>. Speech: <generate_audio kind="speech">words</generate_audio>. Never write code, scripts or links to make media and never explain how to run a generator; just use the tag.)';
// Did the user ask for a picture, video or speech? Only then is the media reminder sent and are media tags acted on,
// so "what can you do?" doesn't make an image.
function asksForMedia(t){t=String(t||'');
  return /\b(image|picture|photo|pic|drawing|painting|artwork|illustration|wallpaper|background|logo|icon|poster|portrait|render|sketch|video|clip|animation|movie)s?\b/i.test(t)&&
         /\b(make|generate|create|draw|paint|render|produce|design|show|give|craft|sketch|animate|film|want|need|like|get|do|can you|could you|another|again|one more)\b/i.test(t)
    ||/\b(draw|paint|sketch|animate)\b/i.test(t)
    ||/\b(say|speak|read)\b[^.?!\n]{0,80}\b(out loud|aloud)\b|\b(text to speech|voice clip|in your voice)\b/i.test(t)}
function apiMessages(msgs,extra,noReminder){const lastUser=[...msgs].reverse().find(m=>m.role=='user');
  const out=[{role:'system',content:systemPrompt(extra,lastUser&&lastUser.content)}].concat(msgs.map(m=>{const o={role:m.role,content:m.role=='assistant'?stripTags(splitThink(m.content).body).trim():m.content};if(m.images&&m.images.length)o.images=m.images;return o}));
  // the media reminder is skipped when the user attached something: then they want it read, not generated
  if(!noReminder&&lastUser&&asksForMedia(lastUser.content)){for(let i=out.length-1;i>0;i--)if(out[i].role=='user'){if(!out[i].images&&!lastUser.docs)out[i]=Object.assign({},out[i],{content:out[i].content+REMINDER});break}}
  return out}
// Recognizes clear requests for media so Ithaca can make them directly instead of hoping the model writes a tag.
function mediaIntent(msgs){
  const last=msgs[msgs.length-1].content;
  const verbs='make|generate|create|draw|paint|render|produce|design|show me|give me|craft|sketch|animate|film|can you (?:make|do|create|generate)';
  const img=new RegExp(`\\b(${verbs})\\b[^.?!\\n]{0,60}\\b(image|picture|photo|pic|drawing|painting|artwork|art|illustration|wallpaper|logo|portrait|poster|icon)s?\\b`,'i');
  const vid=new RegExp(`\\b(${verbs})\\b[^.?!\\n]{0,60}\\b(video|clip|animation|movie|film)s?\\b`,'i');
  const aud=/\b(say|speak|read)\b[^.?!\n]{0,80}\b(out loud|aloud)\b|\b(text to speech|voice clip|in your voice|say it out loud)\b/i;
  const detect=t=>vid.test(t)?'video':img.test(t)?'image':aud.test(t)?'audio':null;
  let kind=detect(last),request=last;
  if(!kind&&/\b(again|retry|try (it )?again|one more|another one|that failed|didn'?t work|still failed|do it)\b/i.test(last)){
    for(let i=msgs.length-2;i>=0&&i>=msgs.length-7;i--)if(msgs[i].role=='user'){const k=detect(msgs[i].content);if(k){kind=k;request=msgs[i].content+'\n'+last;break}}}
  return kind?{kind,request}:null}
async function writeMediaPrompt(it,el,history){
  const what={image:'an image generator',video:'a text-to-video generator',audio:'a text-to-speech voice'}[it.kind];
  const guide=it.kind=='audio'?'Reply with ONLY the exact words to speak, nothing else.'
    :it.kind=='video'?'Write ONE vivid prompt describing a single continuous shot: subject, one clear action, setting, camera movement, lighting and style. Under 80 words. Reply with the prompt ONLY.'
    :'Write ONE vivid, specific prompt: subject, setting, composition, lighting, style and medium. Under 70 words. Reply with the prompt ONLY.';
  const ctx=history.slice(-6).map(m=>`${m.role=='user'?'User':'Assistant'}: ${stripTags(splitThink(m.content).body).slice(0,400)}`).join('\n');
  el.innerHTML='<span class="memchip">'+ico(it.kind=='audio'?'audio':it.kind)+'Planning the '+(it.kind=='audio'?'script':it.kind)+'…</span>';
  const r=await stream([{role:'system',content:`You write inputs for ${what}. Turn the user's request into the input. If they leave the subject up to you, pick something beautiful and specific. ${guide} No quotes, no code, no explanations.`},
    {role:'user',content:`Conversation so far:\n${ctx}\n\nRequest: ${it.request}\n\nWrite the input now. /no_think`}],{max_tokens:220,temperature:0.7});
  let prompt=stripTags(splitThink(r.text).body).replace(/```[a-z]*|```/g,'').replace(/^\s*(prompt|input|text)\s*:\s*/i,'').replace(/^["'“]|["'”]$/g,'').trim();
  el.innerHTML='';
  return prompt||it.request}
function mediaNote(it,prompt){const k=it.kind=='audio'?'audio':it.kind;
  return `# What is happening (written by Ithaca, not the user)\nIthaca's built-in ${it.kind=='audio'?'speech':it.kind} engine is creating the ${k} right now from this ${it.kind=='audio'?'script':'prompt'}: "${prompt}". It will appear directly below your reply when it is ready.\nReply in 1-2 friendly sentences describing what you are making. Do NOT write code, tags, links, markdown images, or instructions for running anything.`}

// Copies a generated image to the clipboard as PNG, so it can be pasted into another chat or app.
window.copyImg=async src=>{
  try{let b=await (await fetch(src)).blob();
    if(b.type!='image/png'){const im=await createImageBitmap(b),c=document.createElement('canvas');c.width=im.width;c.height=im.height;c.getContext('2d').drawImage(im,0,0);b=await new Promise(r=>c.toBlob(r,'image/png'))}
    window._clipImg=new File([b],'image.png',{type:'image/png'});   // in-app copy works even where the system clipboard is blocked
    try{await navigator.clipboard.write([new ClipboardItem({'image/png':b})])}catch(e){}
    toast('Image copied')}
  catch(e){toast('Could not copy: '+e.message)}};
window.viewImg=src=>{const d=document.createElement('div');d.className='lbox';
  d.innerHTML=`<img src="${esc(src)}"><div class="lbar"><a href="${esc(src)}" download>${ico('dl')}Save image</a><button>${ico('x')}Close</button></div>`;
  const close=()=>{d.remove();document.removeEventListener('keydown',k)},k=e=>{if(e.key=='Escape')close()};
  d.onclick=e=>{if(e.target==d||e.target.tagName=='IMG'||e.target.closest('button'))close()};
  document.addEventListener('keydown',k);document.body.appendChild(d)};
/* ---------- attachments: images and files ---------- */
let pend={images:[],docs:[]};
const fmtK=n=>n>=1e6?(n/1e6).toFixed(1)+'M':n>=1e3?Math.round(n/1e3)+'k':String(n);
function canSend(){return !!($('q').value.trim()||pend.images.length||pend.docs.length)}
function drawAtt(){
  const a=$('att'),files=(chat&&chat.files)||[];
  a.innerHTML=pend.images.map((u,i)=>`<div class="thumb"><img src="${u}"><button data-ri="${i}" title="Remove">×</button></div>`).join('')+
    pend.docs.map((d,i)=>`<div class="fchip">${ico('file')}<span>${esc(d.name)}</span><small>${d.busy?'reading…':(d.pages?d.pages+' pages · ':'')+fmtK(d.chars)+' chars'}</small><button data-rd="${i}" title="Remove">×</button></div>`).join('');
  a.style.display=pend.images.length||pend.docs.length?'flex':'none';
  a.querySelectorAll('[data-ri]').forEach(b=>b.onclick=()=>{pend.images.splice(+b.dataset.ri,1);drawAtt()});
  a.querySelectorAll('[data-rd]').forEach(b=>b.onclick=()=>{pend.docs.splice(+b.dataset.rd,1);drawAtt()});
  const k=$('kb');
  k.innerHTML=files.length?`<span>${ico('file')}${files.length} file${files.length>1?'s':''} in this chat:</span>`+files.map(f=>`<span class="kf" title="${esc(f.name)}">${esc(f.name.length>28?f.name.slice(0,26)+'…':f.name)}<button data-rf="${f.id}" title="Stop using this file">×</button></span>`).join(''):'';
  k.style.display=files.length?'flex':'none';
  k.querySelectorAll('[data-rf]').forEach(b=>b.onclick=()=>{chat.files=chat.files.filter(f=>f.id!=b.dataset.rf);drawAtt();saveChat();updateCtx()});
  if(!busy)$('send').disabled=!canSend()}
// Images are shrunk in the browser (the vision model sees 896 px anyway) so chats stay small.
function shrinkImage(file){return new Promise((res,rej)=>{const url=URL.createObjectURL(file),im=new Image();
  im.onload=()=>{const s=Math.min(1,1280/Math.max(im.width,im.height)),c=document.createElement('canvas');c.width=Math.round(im.width*s);c.height=Math.round(im.height*s);
    const g=c.getContext('2d');g.fillStyle='#fff';g.fillRect(0,0,c.width,c.height);g.drawImage(im,0,0,c.width,c.height);URL.revokeObjectURL(url);res(c.toDataURL('image/jpeg',0.92))};
  im.onerror=()=>{URL.revokeObjectURL(url);rej(new Error('not an image'))};im.src=url})}
function fileB64(f){return new Promise(res=>{const r=new FileReader();r.onload=()=>res(String(r.result).replace(/^data:[^,]*,/,''));r.readAsDataURL(f)})}
async function addFiles(list){
  for(const f of list){
    if(f.type.startsWith('image/')){
      if(!status.vision)toast(status.loaded?'The loaded model can\'t see images. Use a vision model like Gemma 3 with its mmproj file next to it.':'Load a vision model first (e.g. Gemma 3).');
      try{pend.images.push(await shrinkImage(f))}catch(e){toast(f.name+': could not read this image')}drawAtt();continue}
    if(f.size>45e6){toast(f.name+' is too large (45 MB max)');continue}
    const d={name:f.name,busy:true};pend.docs.push(d);drawAtt();
    const r=await api('/api/doc/extract',{name:f.name,data:await fileB64(f)});
    if(r.error){toast(f.name+': '+r.error);pend.docs.splice(pend.docs.indexOf(d),1)}
    else Object.assign(d,{busy:false,text:r.text,chars:r.chars,pages:r.pages,kind:r.kind});
    drawAtt()}}
// File search: small files go in whole; big ones are split into passages and ranked (BM25) against the question.
const STOP=new Set('the a an and or of to in on for with is are was were be been it this that these those as at by from what which who how why when where do does did can could would should will you your my me i we our they their them about into than then there here not no yes if so but also any all some just like'.split(' '));
const words=t=>(t.toLowerCase().match(/[\p{L}\p{N}]{2,}/gu)||[]).filter(w=>!STOP.has(w));
const chunkCache=new WeakMap();
function chunksOf(f){
  if(chunkCache.has(f))return chunkCache.get(f);
  const t=f.text,marks=[...t.matchAll(/\[(Page|Slide) (\d+)\]/g)].map(m=>[m.index,m[1]+' '+m[2]]),out=[],size=1400,step=1150;
  for(let i=0;i<t.length;i+=step){let s=i;if(s>0){const sp=t.indexOf(' ',s);if(sp>0&&sp<s+60)s=sp+1}
    const text=t.slice(s,s+size);let where='';for(const [p,l] of marks){if(p<=s+200)where=l;else break}
    out.push({text,where,w:words(text)})}
  chunkCache.set(f,out);return out}
function fileContext(question){
  const files=(chat&&chat.files)||[];if(!files.length)return'';
  const budget=Math.max(3000,Math.floor((status.ctx||4096)*0.42*3.6));   // ~42% of the context window, in characters
  const total=files.reduce((n,f)=>n+f.text.length,0);
  let s=`# Files the user attached\nUse these to answer. Quote or cite the file (and page when shown) like (${files[0].name}${files[0].pages?', page 2':''}). If the answer is not in the files, say so instead of guessing.\n`;
  if(total<=budget){files.forEach(f=>s+=`\n## ${f.name}\n${f.text}\n`);return s}
  const all=[];files.forEach(f=>chunksOf(f).forEach((c,i)=>all.push({f,i,c})));
  const q=words(question);const df={};all.forEach(x=>new Set(x.c.w).forEach(w=>df[w]=(df[w]||0)+1));
  const N=all.length,avg=all.reduce((n,x)=>n+x.c.w.length,0)/Math.max(1,N);
  all.forEach(x=>{const tf={};x.c.w.forEach(w=>tf[w]=(tf[w]||0)+1);let sc=0;
    for(const w of new Set(q)){if(!tf[w])continue;const idf=Math.log(1+(N-df[w]+.5)/(df[w]+.5));sc+=idf*tf[w]*2.2/(tf[w]+1.2*(.25+.75*x.c.w.length/avg))}
    x.score=sc+(x.i==0?0.3:0)});   // a little weight on each file's opening passage
  const pick=[];let used=0;
  for(const x of all.slice().sort((a,b)=>b.score-a.score)){if(used+x.c.text.length>budget)continue;pick.push(x);used+=x.c.text.length;if(used>budget*0.97)break}
  pick.sort((a,b)=>files.indexOf(a.f)-files.indexOf(b.f)||a.i-b.i);
  s+=`The files are long, so these are the passages most relevant to the latest message (${pick.length} of ${N}).\n`;
  pick.forEach(x=>s+=`\n## ${x.f.name}${x.c.where?' · '+x.c.where:''}\n${x.c.text}\n`);
  return s}

/* ---------- streaming ---------- */
async function stream(messages,opts,onText){
  abortCtl=new AbortController();let text='',stats=null,error=null;
  try{const r=await fetch('/api/chat',{method:'POST',signal:abortCtl.signal,body:JSON.stringify(Object.assign({messages,temperature:prefs.temperature,max_tokens:prefs.max_tokens},opts||{}))});
    const rd=r.body.getReader(),dec=new TextDecoder();let buf='';
    for(;;){const {value,done}=await rd.read();if(done)break;buf+=dec.decode(value,{stream:true});let i;
      while((i=buf.indexOf('\n\n'))>=0){const line=buf.slice(0,i);buf=buf.slice(i+2);if(!line.startsWith('data: '))continue;const ev=JSON.parse(line.slice(6));
        if(ev.t){text+=ev.t;onText&&onText(text)}if(ev.error)error=ev.error;if(ev.done)stats=ev.stats}}
  }catch(e){if(e.name!='AbortError')error=String(e)}
  abortCtl=null;return{text,stats,error}}
function stopGen(){fetch('/api/stop',{method:'POST'});stopped=true}
let stopped=false,fromChip=false;

/* ---------- rendering ---------- */
function renderAI(m,live){
  const {think,body,open}=splitThink(m.content||'');let h='';
  if(m.passes&&m.passes.length)h+=`<details class="think"><summary>${ico('brain')}Thought hard · ${m.passes.length==1?'planned':m.passes.length==2?'planned and checked':m.passes.length+' passes'}</summary><div class="tb">`+m.passes.map((p,i)=>`<div class="pass"><h4>${m.passes.length>2?`Pass ${i+1} · `:''}${esc(p.title)}</h4>${md(p.text)}</div>`).join('')+'</div></details>';
  if(think)h+=`<details class="think"${open&&live?' open':''}><summary>${ico('brain')}${open&&live?'Thinking…':'Thought process'}</summary><div class="tb">${md(think)}</div></details>`;
  const chips=(m.tools||[]).concat(m.mem||[]);
  if(chips.length)h+='<div>'+chips.map(o=>`<span class="memchip">${ico(o.i||'mem')}${esc(o.k)}${o.t?': '+esc(o.t.length>70?o.t.slice(0,70)+'…':o.t):''}</span>`).join('')+'</div>';
  // the picture sits where the model wrote its tag: text before it, the picture (or its live card), then the text after it
  const tm=body.match(/<generate_(image|video|audio)\b[\s\S]*?(?:<\/generate_\1>|\/>)/);
  const before=stripTags(tm?body.slice(0,tm.index):body),after=tm?stripTags(body.slice(tm.index+tm[0].length)):'';
  h+=md(before);
  const mh=(m.media||[]).map(mediaHtml).join('');
  h+=mh||(live&&m.pendingMedia?'<div class="mslot"></div>':'');
  if(after.trim())h+=md(after);
  if(live&&!before.trim()&&!after.trim()&&!think&&!m.passes&&!mh&&!m.pendingMedia)
    h+=`<span class="waiting"><span class="dots" style="--ph:-${Math.round(performance.now()%1000)}ms"><span></span><span></span><span></span></span><span class="wl" style="${phase(1800)}">${esc(m.wait||'')}</span></span>`;
  if(m.error)h+=`<div class="err">${esc(m.error)}</div>`;
  return h}
function mediaHtml(x){return x.url?(x.kind=='image'?`<div class="gwrap${justMade.has(x.url)?' justmade':''}"><img class="genimg" src="${esc(x.url)}" onclick="viewImg(this.src)" title="Click to view"><div class="gbtns"><button onclick="copyImg('${esc(x.url)}')" title="Copy image">${ico('copy')}</button><button onclick="viewImg('${esc(x.url)}')" title="View full size">${ico('expand')}</button><a href="${esc(x.url)}" download title="Save image">${ico('dl')}</a></div></div>`:x.kind=='video'?`<video class="genimg" controls src="${esc(x.url)}"></video>`:`<audio controls src="${esc(x.url)}"></audio>`):''}
const justMade=new Set();   // pictures that finished during this reply get the reveal once
// Animations keep their phase when the element is rebuilt (replies re-render on every token), so nothing restarts.
const phase=ms=>`animation-delay:-${Math.round(performance.now()%ms)}ms`;
// The live render card is created once and moved into each re-render's slot.
let liveCard=null;
const chunkBorn=new WeakMap();   // reply object -> birth time of each top-level block
function fadeChunks(el,ai){let born=chunkBorn.get(ai);if(!born){born=[];chunkBorn.set(ai,born)}const now=performance.now();
  [...el.children].forEach((c,i)=>{if(c.classList.contains('mslot')||c.classList.contains('waiting')||c.querySelector('.rcard'))return;
    if(born[i]==null)born[i]=now;const age=now-born[i];if(age<380){c.classList.add('chunk');c.style.animationDelay=`-${Math.round(age)}ms`}})}
function paint(el,ai){el.innerHTML=renderAI(ai,true);fadeChunks(el,ai);
  if(liveCard){const s=el.querySelector('.mslot');if(s){s.appendChild(liveCard);
    // moving a node restarts its CSS animations; restart them at the same point of the shared clock
    liveCard.querySelectorAll('.rshimmer').forEach(x=>x.style.animationDelay=`-${Math.round(performance.now()%1600)}ms`)}}}
function msgEl(m,i,isLast){
  const d=document.createElement('div');
  if(m.role=='user'){d.className='msg user';d.innerHTML=(m.images&&m.images.length?`<div class="uimgs">${m.images.map(u=>`<img src="${u}" onclick="window.open().document.write('<img src=&quot;'+this.src+'&quot; style=&quot;max-width:100%&quot;>')">`).join('')}</div>`:'')+
    (m.docs&&m.docs.length?`<div class="udocs">${m.docs.map(x=>`<div class="fchip">${ico('file')}<span>${esc(x.name)}</span><small>${x.pages?x.pages+' pages':fmtK(x.chars)+' chars'}</small></div>`).join('')}</div>`:'')+
    `<div class="bubble">${esc(m.content)}</div><div class="actions"><button title="Edit" onclick="editMsg(${i})">${ico('edit')}</button><button title="Copy" onclick="copyMsg(${i})">${ico('copy')}</button></div>`}
  else{d.className='msg ai'+(isLast?' last':'');const s=m.stats;
    d.innerHTML=`<div class="avatar"><svg><use href="#spark"/></svg></div><div class="body"><div class="content">${renderAI(m)}</div><div class="actions"><button title="Copy" onclick="copyMsg(${i})">${ico('copy')}</button>`+
      (isLast?`<button title="Regenerate" onclick="redo()">${ico('redo')}</button>`:'')+(s&&s.tps?`<span class="meta">${s.gen} tokens · ${s.tps} tok/s${s.reused?` · ${s.reused} cached`:''}</span>`:'')+'</div></div>'}
  return d}
function render(){
  const log=$('log');
  if(!chat||!chat.messages.length){
    const hi=prefs.name?`Hi ${esc(prefs.name)}, what's on your mind?`:'How can I help you today?';
    log.innerHTML=`<div class="empty"><div class="logo" style="width:46px;height:46px;border-radius:14px"><svg style="width:26px;height:26px"><use href="#spark"/></svg><i class="spark" style="left:-10px;top:4px;animation-delay:1s"></i><i class="spark" style="right:-8px;top:-6px;animation-delay:1.6s"></i><i class="spark" style="right:-12px;bottom:2px;animation-delay:2.3s"></i></div><h2>${hi}</h2><div class="chips">`+
      ['Explain how attention works in transformers','Write a Python script that renames files by date','Help me plan a Blender animation workflow','What do you remember about me?'].map(t=>`<button class="chip">${t}</button>`).join('')+'</div></div>';
    log.querySelectorAll('.chip').forEach(c=>c.onclick=()=>{$('q').value=c.textContent;fromChip=true;send()});updateCtx();drawAtt();return}
  const col=document.createElement('div');col.className='col';
  if(chat.summary)col.innerHTML=`<details class="think"><summary>${ico('compact')}Earlier messages were compacted into a summary</summary><div class="tb">${md(chat.summary)}</div></details>`;
  chat.messages.forEach((m,i)=>{const e=msgEl(m,i,i==chat.messages.length-1&&m.role=='assistant');if(i>=freshFrom)e.classList.add('fresh');col.appendChild(e)});
  freshFrom=1e9;
  log.innerHTML='';log.appendChild(col);log.scrollTop=log.scrollHeight;updateCtx();drawAtt()}
window.copyMsg=i=>{const m=chat.messages[i];navigator.clipboard.writeText(m.role=='assistant'?stripTags(splitThink(m.content).body).trim():m.content);toast('Copied')};

/* ---------- chats ---------- */
async function loadHistory(){chats=await api('/api/chats');drawHistory()}
let knownChats=null;
function drawHistory(){
  const q=$('search').value.toLowerCase();const now=Date.now(),day=864e5;const groups={};
  chats.filter(c=>!q||c.title.toLowerCase().includes(q)).forEach(c=>{const age=now-c.updated;const g=age<day?'Today':age<2*day?'Yesterday':age<7*day?'Previous 7 days':'Older';(groups[g]=groups[g]||[]).push(c)});
  $('history').innerHTML=Object.entries(groups).map(([g,list])=>`<div class="group">${g}</div>`+list.map(c=>`<div class="hitem${chat&&chat.id==c.id?' on':''}" data-id="${c.id}"><span>${esc(c.title)}</span><button class="x" data-del="${c.id}" title="Delete">${ico('trash')}</button></div>`).join('')).join('')||'<div class="group">No chats yet</div>';
  const ids=new Set(chats.map(c=>c.id));
  if(knownChats)$('history').querySelectorAll('.hitem').forEach(e=>{if(!knownChats.has(e.dataset.id))e.classList.add('slidein')});
  knownChats=ids;
  $('history').querySelectorAll('.hitem').forEach(e=>e.onclick=ev=>{if(ev.target.closest('[data-del]'))return;openChat(e.dataset.id)});
  $('history').querySelectorAll('[data-del]').forEach(b=>b.onclick=async()=>{if(!confirm('Delete this chat?'))return;await api('/api/chats/'+b.dataset.del+'/delete',{});if(chat&&chat.id==b.dataset.del)newChat();loadHistory()})}
async function openChat(id){if(busy)return;const c=await api('/api/chats/'+id);if(c.error)return;chat=c;if(view!='chat')setView('chat');render();drawHistory()}
function newChat(){if(busy)return;if(typeof view!='undefined'&&view!='chat')setView('chat');chat={id:uid(),title:'New chat',created:Date.now(),updated:Date.now(),messages:[],summary:''};render();drawHistory();$('q').focus()}
async function saveChat(){if(!chat||!chat.messages.length)return;chat.updated=Date.now();
  if(chat.title=='New chat'){const f=chat.messages[0].content.replace(/\s+/g,' ').trim();chat.title=f.length>48?f.slice(0,48)+'…':f}
  await api('/api/chats/'+chat.id,chat);loadHistory()}

/* ---------- context meter ---------- */
async function updateCtx(){
  if(!status.loaded){$('ctxText').textContent='no model';$('ctxBar').firstChild.style.width='0';$('compact').disabled=true;return}
  const r=await api('/api/count',{messages:apiMessages(chat?chat.messages:[])});const pct=r.ctx?r.tokens/r.ctx:0;
  $('ctxText').textContent=`${r.tokens.toLocaleString()} / ${r.ctx.toLocaleString()} tokens · ${Math.max(0,100-Math.round(pct*100))}% left`;
  const b=$('ctxBar');b.firstChild.style.width=Math.min(100,pct*100)+'%';b.className='ctxbar'+(pct>.9?' bad':pct>.7?' warn':'');
  $('compact').disabled=busy||!chat||chat.messages.length<4}

/* ---------- sending ---------- */
// Think hard: one short planning pass; a checking pass only when the plan is unsure or the task is tricky (code, maths, long).
const PLAN=['Plan','In under 150 words of bullet points: what is really being asked, the key facts, steps or code structure the answer needs, and the one mistake most likely to sneak in. End with one line: CONFIDENCE: high, medium or low (how sure you are this plan is right).'];
const CHECK=['Check','Review the plan below like a skeptical expert, in under 120 words: fix anything wrong, add what is missing, re-check every number, step and line of code. End with the corrected key points.'];
function setBusy(b){const changed=busy!=b;busy=b;const s=$('send');s.innerHTML=ico(b?'stop':'up');s.classList.toggle('busy',b);
  if(changed){s.classList.remove('morph');void s.offsetWidth;s.classList.add('morph')}s.disabled=!b&&!canSend();s.title=b?'Stop':'Send';updateCtx()}
async function runTools(tools){const results=[],chips=[],media=[];
  for(const t of tools){
    if(t.tag=='memory_search'){const r=searchMem(t.val);
      results.push(`memory_search("${t.val}") → `+(r.length?'\n'+r.map(memLine).join('\n'):'no matching memories'));chips.push({i:'mem',k:'Searched memory',t:`"${t.val}" · ${r.length} found`})}
    else{const kind=t.tag.slice(9);const holder=document.querySelector('.msg.ai:last-child .content');
      liveCard=document.createElement('div');if(holder)(holder.querySelector('.mslot')||holder).appendChild(liveCard);const stop=watchProgress(liveCard);
      // tag attributes (like kind="speech") must not override which engine is called; images get a size that fits the request
      const size=kind=='image'?Object.assign(pickSize(t.a,lastRequest),{fit:true}):{};
      const r=await api('/api/media/generate',Object.assign({},t.a,size,{kind,prompt:t.val,mode:t.a.kind||t.a.mode||''}));stop();liveCard=null;
      results.push(`${t.tag} → `+(r.error?'FAILED: '+r.error:`done. The ${kind} is already displayed to the user right below your message, so do not add links, file names or markdown images; just describe it briefly or offer changes.`));chips.push({i:kind=='audio'?'audio':kind=='video'?'video':'image',k:`${kind[0].toUpperCase()+kind.slice(1)} request`,t:r.error?'not available':'created'});
      if(r.url){media.push({kind,url:r.url});justMade.add(r.url);setTimeout(()=>justMade.delete(r.url),2500)}}}
  return{results,chips,media}}
// Width and height for an image: from the tag (width/height, size="WxH", aspect="16:9"), else from the words of the request.
// The engine keeps the shape and renders at the model's native pixel count.
let lastRequest='';
function pickSize(a,req){let w=+a.width||0,h=+a.height||0;
  const m=String(a.size||a.resolution||a.aspect||a.ratio||'').match(/(\d+(?:\.\d+)?)\s*[x×:*]\s*(\d+(?:\.\d+)?)/);if(!(w>0&&h>0)&&m){w=+m[1];h=+m[2]}
  if(!(w>0&&h>0)){const t=String(req||'').toLowerCase(),e=t.match(/(\d{3,4})\s*[x×*]\s*(\d{3,4})/),q=t.match(/\b(\d{1,2})\s*:\s*(\d{1,2})\b/);
    if(e){w=+e[1];h=+e[2]}else if(q){w=+q[1];h=+q[2]}
    else if(/\b(iphone|phone|mobile|lock ?screen|android|story|stories|tiktok|reels?|vertical)\b/.test(t)){w=9;h=19.5}
    else if(/\b(portrait|poster|book cover|pinterest|tall)\b/.test(t)){w=2;h=3}
    else if(/\b(desktop|monitor|pc|laptop|youtube|banner|thumbnail|widescreen|landscape|cinematic|header|wallpaper|background|panorama)\b/.test(t)){w=16;h=9}
    else{w=1;h=1}}
  return{width:Math.round(w*1000),height:Math.round(h*1000)}}
async function generate(){
  if(!status.loaded&&!status.paused){toast('Load a model first (Settings → Model)');openSettings('model');return}
  setBusy(true);stopped=false;
  const ai={role:'assistant',content:''};chat.messages.push(ai);render();if(flyFrom)flyLogo();
  const el=$('log').querySelector('.col').lastChild.querySelector('.content');
  const history=chat.messages.slice(0,-1);let extra='';
  lastRequest=history[history.length-1].content;
  // while nothing has been written yet, say whether the model is still loading (after an image, or a fresh load)
  const loadingNow=()=>status.paused||!status.loaded||(status.layers||[]).some(l=>l[1]!=l[0]);
  ai.wait=loadingNow()?'Loading model…':'';
  const waitTimer=setInterval(()=>{const w=el.querySelector('.wl');if(!w)return;const t=loadingNow()?'Loading model…':'';
    if(t!=w.textContent){ai.wait=t;w.textContent=t;w.style.animationDelay=`-${Math.round(performance.now()%1800)}ms`}},500);
  if(hard){
    // Build the animated header once; only the notes area below it re-renders while tokens stream in.
    ai.passes=[];
    el.innerHTML=`<div class="hard"><div class="orbit"><i></i><i></i><i></i></div><div class="hardtxt"><b>Thinking hard…</b><span class="hlbl"></span></div><div class="steps"><i></i></div></div><div class="live"></div>`;
    const lbl=el.querySelector('.hlbl'),stepsEl=el.querySelector('.steps'),live=el.querySelector('.live');
    const q=history[history.length-1].content;
    const pass=async([title,instr],max)=>{
      const n=ai.passes.length;lbl.textContent=title=='Plan'?'Making a plan':'Checking the plan';
      stepsEl.querySelectorAll('i').forEach((x,i)=>x.className=i<n?'done':i==n?'cur':'');
      const notes=ai.passes.map(x=>`## ${x.title}\n${x.text}`).join('\n\n');
      const msgs=apiMessages(history.slice(0,-1).concat([{role:'user',content:`${q}\n\n---\n[Private thinking step: ${title}. Working notes only: do NOT write the final answer and do NOT use any tags.]\n${instr}`+(notes?`\n\nThe plan so far:\n${notes}`:'')+' /no_think'}]),'',true);
      const r=await stream(msgs,{max_tokens:max,temperature:0.3},t=>{const x=splitThink(t);live.innerHTML=renderAI({passes:ai.passes.concat([{title:title+' (writing…)',text:stripTags(x.body||x.think)}])})});
      const x=splitThink(r.text),text=stripTags(x.body||x.think).trim();if(r.error)ai.error=r.error;return text};
    const plan=await pass(PLAN,450);
    const sure=/CONFIDENCE:\s*\**\s*high/i.test(plan);
    ai.passes.push({title:'Plan',text:plan.replace(/\n?.*CONFIDENCE:.*$/i,'').trim()});
    const tricky=q.length>500||/```|\b(code|function|script|program|bug|error|debug|regex|sql|calculate|compute|prove|equation|math|probability|step[- ]by[- ]step)\b|\d+\s*[-+*\/^%]\s*\d+/i.test(q)||(q.match(/\d+(?:[.,:]\d+)?/g)||[]).length>=2;
    if(!stopped&&!ai.error&&(!sure||tricky)){stepsEl.insertAdjacentHTML('beforeend','<i></i>');ai.passes.push({title:'Check',text:await pass(CHECK,350)})}
    stepsEl.querySelectorAll('i').forEach(x=>x.className='done');
    extra=`# Your private notes for the latest message\nYou already planned this answer. Use the notes to write the best possible final answer. Do not mention the notes.\n\n`+ai.passes.map(x=>`## ${x.title}\n${x.text}`).join('\n\n');}
  const lastU=history[history.length-1];
  const intent=hard||(lastU.images&&lastU.images.length)||(lastU.docs&&lastU.docs.length)?null:mediaIntent(history);
  let mediaPrompt=null;
  if(intent&&!stopped){mediaPrompt=await writeMediaPrompt(intent,el,history);extra=mediaNote(intent,mediaPrompt)}
  if(!stopped&&!ai.error){
    const r=await stream(apiMessages(history,extra,!!intent),{},t=>{ai.content=t;paint(el,ai);scrollDown()});
    ai.content=r.text;ai.stats=r.stats;if(r.error)ai.error=r.error;
    // the reply text comes first; then the media renders live underneath it
    if(mediaPrompt&&!stopped){ai.content=splitThink(ai.content).body.trim().split(/\n\s*(?:-{3,}|\n)/)[0].trim();ai.pendingMedia=true;paint(el,ai);   // small models ramble after the intro
      const tr=await runTools([{tag:'generate_'+intent.kind,a:intent.kind=='audio'?{kind:'speech'}:{},val:mediaPrompt}]);ai.pendingMedia=false;
      ai.tools=(ai.tools||[]).concat(tr.chips);ai.media=(ai.media||[]).concat(tr.media);
      if(!tr.media.length)ai.content+='\n\n*Sorry, that didn\'t work: '+tr.results.join(' ').replace(/^.*FAILED:\s*/,'')+'*';
      paint(el,ai)}
    // Tool loop: memory searches feed results back so the model can finish its reply. A media tag is made once, where the
    // model wrote it; the text around it is already the reply, so there is no follow-up round for it.
    // One picture per reply unless the user asked for several.
    const several=/\b(two|three|four|five|six|\d+|several|multiple|a few|some|variations?|versions?)\b[^.?!\n]{0,40}\b(images|pictures|photos|pics|wallpapers|drawings|videos|clips)\b/i.test(lastU.content);
    let done=0,made=(ai.media||[]).length;const asked=asksForMedia(lastU.content);
    for(let round=0;round<3&&!stopped&&!ai.error;round++){
      let calls=parseTools(ai.content).filter(t=>(!t.tag.startsWith('memory_')||t.tag=='memory_search')&&!((intent||(lastU.images&&lastU.images.length)||(lastU.docs&&lastU.docs.length))&&t.tag.startsWith('generate_'))).slice(done);
      if(!calls.length)break;done+=calls.length;
      calls=calls.filter(t=>!t.tag.startsWith('generate_')||(asked&&(several||made++==0)));
      if(!calls.length)break;
      const media=calls.some(t=>t.tag.startsWith('generate_'));
      if(media){ai.pendingMedia=true;paint(el,ai)}
      const tr=await runTools(calls);ai.pendingMedia=false;ai.tools=(ai.tools||[]).concat(tr.chips);ai.media=(ai.media||[]).concat(tr.media);
      if(media&&!tr.media.length)ai.content+='\n\n*Sorry, that didn\'t work: '+tr.results.filter(x=>/FAILED/.test(x)).join(' ').replace(/^.*FAILED:\s*/,'')+'*';
      paint(el,ai);
      if(!calls.some(t=>t.tag=='memory_search'))break;
      const prev=ai.content;
      const msgs=apiMessages(history.concat([{role:'assistant',content:prev},{role:'user',content:'[Tool results, sent automatically, not written by the user]\n'+tr.results.join('\n')+'\n\nContinue your reply to the user using these results. Do not repeat what you already wrote and do not call the same tool again.'}]),extra);
      const r2=await stream(msgs,{},t=>{ai.content=prev+'\n\n'+splitThink(t).body;paint(el,ai);scrollDown()});
      ai.content=prev+'\n\n'+splitThink(r2.text).body;if(r2.stats)ai.stats=r2.stats;if(r2.error)ai.error=r2.error}}
  if(prefs.memory){let ops=applyMemoryOps(parseTools(ai.content));
    const u=history[history.length-1].content;
    // Backup: models often say "I'll remember" without writing the tag, so run a short focused extraction pass.
    // (questions like "what do you remember about me?" and the starter suggestions are not facts to save)
    const question=/\?\s*$/.test(u)||/^\s*(what|who|how|why|when|where|which|do|does|did|can|could|would|should|is|are)\b/i.test(u);
    if(!ops.length&&!stopped&&!fromChip&&!question&&/\b(remember|forget|my name|call me|i'?m |i am |i use|i work|i prefer|i like|i love|i hate|i don'?t like|my (job|project|pc|computer|gpu|wife|husband|kid|dog|cat)|i live|i'?ve been)/i.test(u)){
      const r=await stream([{role:'system',content:`You extract long-term memories about the user from their latest message. Existing memories:\n${memories.map(memLine).join('\n')||'(none)'}\n\nOutput ONLY tags, one per line, for lasting facts (identity, preferences, tools, projects, explicit "remember"/"forget" requests):\n<memory_save category="about|preferences|projects|knowledge|other">User ...</memory_save>\n<memory_edit id="N">full corrected text</memory_edit>\n<memory_delete id="N"/>\nEdit an existing memory instead of duplicating it. Ignore small talk, one-off tasks and secrets. Output nothing if there is nothing worth remembering.`},
        {role:'user',content:u+' /no_think'}],{max_tokens:300,temperature:0});
      ops=applyMemoryOps(parseTools(r.text))}
    if(ops.length)ai.mem=ops}
  // small models tack on "I'll remember that" even when nothing was saved
  if(!(ai.mem&&ai.mem.length))ai.content=ai.content.replace(/\n*[ \t]*(?:Got it[,!.]?\s*)?I(?:'|’)?ll (?:remember|keep) (?:that|this)(?: in mind)?[.!]?\s*$/i,'');
  if(!ai.content&&!ai.error&&stopped)ai.error='Stopped.';
  clearInterval(waitTimer);delete ai.wait;delete ai.pendingMedia;fromChip=false;
  setBusy(false);render();saveChat();refreshStatus()}
function scrollDown(){const l=$('log');if(l.scrollHeight-l.scrollTop-l.clientHeight<200)l.scrollTop=l.scrollHeight}
async function send(){
  if(busy){stopGen();return}
  const q=$('q').value.trim();if(!canSend())return;
  if(pend.docs.some(d=>d.busy)){toast('Still reading your file…');return}
  if(pend.images.length&&!status.vision){toast('The loaded model can\'t see images. Load a vision model such as Gemma 3 (with its mmproj file in the same folder).');return}
  if(!chat)newChat();
  const m={role:'user',content:q||(pend.images.length?'What is in this image?':'Summarize the attached file'+(pend.docs.length>1?'s':'')+'.')};
  if(pend.images.length)m.images=pend.images;
  if(pend.docs.length){chat.files=(chat.files||[]).concat(pend.docs.map(d=>({id:uid(),name:d.name,text:d.text,chars:d.chars,pages:d.pages,kind:d.kind})));
    m.docs=pend.docs.map(d=>({name:d.name,chars:d.chars,pages:d.pages}))}
  pend={images:[],docs:[]};drawAtt();
  const empty=$('log').querySelector('.empty');
  if(empty){flyFrom=empty.querySelector('.logo').getBoundingClientRect();empty.classList.add('leaving');await new Promise(r=>setTimeout(r,330))}
  freshFrom=chat.messages.length;
  $('q').value='';autosize();chat.messages.push(m);render();await generate()}
// The new-chat logo hops in an arc and lands as the first reply's avatar.
let flyFrom=null,freshFrom=1e9;
function flyLogo(){const from=flyFrom;flyFrom=null;const to=document.querySelector('.msg.ai:last-child .avatar');if(!from||!to)return;
  const t=to.getBoundingClientRect(),f=document.createElement('div');f.className='flylogo';f.innerHTML='<svg><use href="#spark"/></svg>';
  Object.assign(f.style,{left:from.left+'px',top:from.top+'px',width:from.width+'px',height:from.height+'px'});document.body.appendChild(f);to.style.visibility='hidden';
  const dx=t.left-from.left+(t.width-from.width)/2,dy=t.top-from.top+(t.height-from.height)/2,k=t.width/from.width;
  const peak=Math.max(Math.min(0,dy*.5)-Math.min(140,Math.abs(dy)*.35+60),24-from.top);   // arc top stays inside the window
  const lift=Math.min(0,dy*.5)-peak;
  const a=f.animate([{transform:'translate(0,0) scale(1) rotate(0)',borderRadius:'14px'},
    {transform:`translate(${dx*.5}px,${peak}px) scale(${(1+k)/2+.15}) rotate(200deg)`,borderRadius:'40%',offset:.5},
    {transform:`translate(${dx}px,${dy}px) scale(${k}) rotate(360deg)`,borderRadius:'50%'}],{duration:720,easing:'cubic-bezier(.45,0,.25,1)'});
  a.onfinish=()=>{f.remove();to.style.visibility='';to.classList.add('land')}}
window.redo=async()=>{if(busy||!chat)return;if(chat.messages[chat.messages.length-1].role=='assistant')chat.messages.pop();await generate()};
window.editMsg=i=>{if(busy)return;const m=chat.messages[i];const d=$('log').querySelector('.col').children[i+(chat.summary?1:0)];
  d.innerHTML=`<textarea class="editbox">${esc(m.content)}</textarea><div class="editbtns"><button class="btn" id="ecancel">Cancel</button><button class="btn pri" id="esave">Save &amp; resend</button></div>`;
  const ta=d.querySelector('textarea');ta.focus();ta.setSelectionRange(ta.value.length,ta.value.length);
  $('ecancel').onclick=render;$('esave').onclick=async()=>{const v=ta.value.trim();if(!v)return;chat.messages=chat.messages.slice(0,i);chat.messages.push(Object.assign({},m,{content:v}));render();await generate()}};

/* ---------- compact ---------- */
$('compact').onclick=async()=>{
  if(busy||!chat||chat.messages.length<4)return;
  const keep=2,old=chat.messages.slice(0,-keep);setBusy(true);toast('Compacting conversation…');
  const transcript=old.map(m=>`${m.role=='user'?'User':'Assistant'}: ${m.role=='assistant'?stripTags(splitThink(m.content).body):m.content}`).join('\n\n');
  const r=await stream([{role:'system',content:'You compress conversations so they can continue in a smaller context window. Write a dense summary in short bullet points, grouped under: Goal, Facts and decisions, Code and files (keep exact names, signatures and key snippets), Open questions / next steps. Preserve every name, number, requirement and preference. Do not add commentary or anything that was not in the conversation.'},
    {role:'user',content:(chat.summary?'Existing summary:\n'+chat.summary+'\n\n':'')+'Conversation to compress:\n\n'+transcript}],{max_tokens:1200,temperature:0.2});
  const s=stripTags(splitThink(r.text).body).trim();
  if(s&&!r.error){chat.summary=s;chat.messages=chat.messages.slice(-keep);await saveChat();toast('Compacted: older messages replaced by a summary')}else toast('Compact failed');
  setBusy(false);render()};

/* ---------- status / tier ring ---------- */
function drawSpec(){const e=$('specState');if(!e)return;const d=status.draft||'off';
  e.textContent=d.startsWith('on: ')?'Draft model in use: '+d.slice(4):d=='off'?'Off.':d.charAt(0).toUpperCase()+d.slice(1)+'.';
  if(d.startsWith('no matching'))e.textContent+=' Download a small model from the same family (for example Qwen3 0.6B for Qwen3 models) to enable it.'}
async function refreshStatus(){
  try{status=await api('/api/status')}catch(e){return}
  drawSpec();
  $('modelName').textContent=status.loaded?status.name:status.paused?'Paused for media':'No model loaded';
  const svg=$('ringSvg');
  if(!status.loaded){svg.innerHTML='<circle cx="18" cy="18" r="15" fill="none" stroke="var(--line)" stroke-width="4"/>';$('ringText').textContent=status.paused?'paused while media is made':'no model loaded';$('ringTip').innerHTML=esc(status.gpu||'');return}
  const L=status.layers.concat([status.out]),n=L.length,C=2*Math.PI*15,seg=C/n,gap=n>40?0.35:0.8;
  const col=t=>t==2?'var(--gpu)':t==1?'var(--ram)':'var(--disk)';
  svg.innerHTML='<circle cx="18" cy="18" r="15" fill="none" stroke="var(--line)" stroke-width="4"/>'+L.map((l,i)=>`<circle cx="18" cy="18" r="15" fill="none" stroke="${col(l[1])}" stroke-width="4" stroke-dasharray="${Math.max(.2,seg-gap)} ${C}" stroke-dashoffset="${-i*seg}" opacity="${l[1]==l[0]?1:.45}"/>`).join('');
  const cnt=t=>L.filter(l=>l[1]==t).length,moving=L.filter(l=>l[1]!=l[0]).length;
  const st=status.stats||{};
  $('ringText').textContent=`${cnt(2)} VRAM · ${cnt(1)} RAM · ${cnt(0)} SSD`+(moving?` · moving ${moving}…`:'')+(st.tps?` · ${st.tps} tok/s`:'');
  $('ringTip').innerHTML=`<b style="color:var(--text)">${esc(status.name)}</b> (${esc(status.arch)})<br>${esc(status.gpu)}<div class="bars">`+
    L.map((l,i)=>`<i style="background:${col(l[1])};opacity:${l[1]==l[0]?1:.45}" title="${i<n-1?'layer '+i:'output head'}"></i>`).join('')+
    `</div>VRAM ${status.gpu_mb} MB · pinned RAM ${status.ram_mb} MB<br>opened in ${status.load_ms} ms · ${status.threads} CPU threads · context ${status.ctx}`+
    (st.tps?`<br>last reply: ${st.tps} tok/s · prompt ${st.prefill_tps} tok/s`:'')+'<br><span style="color:var(--gpu)">■</span> VRAM <span style="color:var(--ram)">■</span> RAM <span style="color:var(--disk)">■</span> SSD stream'}

/* ---------- settings ---------- */
function savePrefs(){api('/api/prefs',prefs);document.documentElement.dataset.theme=prefs.theme=='system'?(matchMedia('(prefers-color-scheme: light)').matches?'light':'dark'):prefs.theme}
function sw(key,title,desc){return `<div class="switch"><div><b>${title}</b><span>${desc}</span></div><button class="sw${prefs[key]?' on':''}" data-sw="${key}"></button></div>`}
function bindPane(){
  $('pane').querySelectorAll('[data-sw]').forEach(b=>b.onclick=()=>{prefs[b.dataset.sw]=!prefs[b.dataset.sw];b.classList.toggle('on');savePrefs();if(b.dataset.sw=='run_code')render();
    if(b.dataset.sw=='spec_decode'){api('/api/spec',{on:prefs.spec_decode});setTimeout(drawSpec,300)}});
  $('pane').querySelectorAll('[data-pref]').forEach(e=>e.onchange=()=>{const k=e.dataset.pref;prefs[k]=e.type=='number'?+e.value:e.value;savePrefs()})}
const TABS={
general:()=>`<h2>General</h2><p class="sub">How Ithaca talks to you.</p>
 <div class="grid2"><div class="field"><label>What should Ithaca call you?</label><input data-pref="name" value="${esc(prefs.name)}" placeholder="Your name"></div>
 <div class="field"><label>Theme</label><select data-pref="theme">${['dark','light','system'].map(t=>`<option ${prefs.theme==t?'selected':''}>${t}</option>`).join('')}</select></div></div>
 <div class="field"><label>About you</label><textarea data-pref="about" rows="2" placeholder="e.g. I'm a 3D artist learning C++.">${esc(prefs.about)}</textarea></div>
 <div class="field"><label>Response style</label><input data-pref="style" value="${esc(prefs.style)}" placeholder="e.g. concise, friendly, use bullet points"></div>
 <div class="field"><label>Custom instructions</label><textarea data-pref="instructions" rows="3" placeholder="Anything Ithaca should always keep in mind">${esc(prefs.instructions)}</textarea></div>
 <div class="grid2"><div class="field"><label>Creativity (temperature)</label><input data-pref="temperature" type="number" step="0.1" min="0" max="2" value="${prefs.temperature}"><small>0 = precise, 1+ = creative</small></div>
 <div class="field"><label>Max reply length (tokens)</label><input data-pref="max_tokens" type="number" step="256" value="${prefs.max_tokens}"></div></div>
 `,
model:()=>`<h2>Model &amp; hardware</h2><p class="sub">Pick a model and how Ithaca spreads it across VRAM, RAM and SSD.</p>
 <div class="field"><label>Model</label><select id="mSel"><option>loading…</option></select></div>
 <div class="field"><label>Add a folder with .gguf files</label><div style="display:flex;gap:8px"><input id="mDir" placeholder="D:\\AI\\models"><button class="btn" id="mAdd">Add</button></div></div>
 <div class="grid2"><div class="field"><label>CPU threads</label><input id="mThreads" type="number" min="0" value="${status.opts?status.opts.threads:0}"><small>0 = auto (half your cores)</small></div>
 <div class="field"><label>Context window (tokens)</label><input id="mCtx" type="number" step="1024" value="${status.opts?status.opts.ctx:4096}"><small>How much the model can keep in mind</small></div>
 <div class="field"><label>VRAM budget (MB)</label><input id="mVram" type="number" value="${status.opts?status.opts.vram_mb:-1}"><small>-1 = auto (leaves ~768 MB free), 0 = CPU only</small></div>
 <div class="field"><label>RAM budget (MB)</label><input id="mRam" type="number" value="${status.opts?status.opts.ram_mb:-1}"><small>-1 = auto (leaves ~2.5 GB free)</small></div></div>
 <button class="btn pri" id="mLoad">Load model</button> <span id="mMsg" style="margin-left:10px;color:var(--dim);font-size:13px"></span>
 <div style="margin-top:18px">${sw('spec_decode','Faster replies (speculative decoding)','A small model with the same vocabulary drafts a few words at a time and the main model checks them in one pass. Answers stay the same; code and structured text often come out 30–60% faster. Ithaca picks the small model by itself and pauses it when it doesn\'t help.')}
 <p class="sub" id="specState" style="margin-top:8px"></p></div>`,
memory:()=>`<h2>Memory</h2><p class="sub">What Ithaca remembers across all chats. The model saves, updates, searches and forgets memories by itself while memory is on. Pinned memories are always shown to the model; with more than 40 memories, it sees the pinned ones plus those relevant to your message and can search the rest.</p>
 ${sw('memory','Use memory','Let the model read, write, search and edit memories while chatting.')}
 <div style="display:flex;gap:8px;margin:16px 0;flex-wrap:wrap"><input id="memFind" class="search" style="margin:0;flex:1;min-width:160px" placeholder="Search memories">
 <select id="memCat" class="search" style="margin:0;width:auto"><option value="">All categories</option>${Object.entries(CATS).map(([k,v])=>`<option value="${k}">${v}</option>`).join('')}</select>
 <button class="btn" id="memAdd">+ Add</button><button class="btn" id="memReview">Review &amp; clean up with AI</button><button class="btn" id="memClear">Delete all</button></div>
 <div id="memList"></div>`,
code:()=>`<h2>Coding</h2><p class="sub">Code blocks get syntax highlighting, copy buttons, and can be run or previewed.</p>
 ${sw('run_code','Run Python blocks','Show a Run button on Python code blocks. Code runs on this PC with your installed Python (30 s limit) only when you click Run.')}
 <p class="sub" style="margin-top:14px">HTML and SVG blocks get a Preview button that renders them in a sandboxed frame.</p>`,
about:()=>`<h2>About Ithaca</h2><p class="sub">A from-scratch local LLM engine and chat app in one small exe. Here's how it compares to the two projects that inspired it.</p>
 <table class="cmp"><tr><th></th><th>Ithaca (this)</th><th>Strata</th><th>Odysseus</th></tr>
 <tr><td>What it is</td><td>Inference engine + chat UI in one exe</td><td>Inference app built to run one huge MoE model on consumer PCs</td><td>Self-hosted AI workspace (chat, agents, research, email, docs)</td></tr>
 <tr><td>Engine</td><td>Own engine: GGUF loader, tokenizer, AVX2 + hand-written PTX GPU kernels</td><td>Builds on parts of llama.cpp / ggml</td><td>Uses external backends (Ollama, llama.cpp, vLLM) or cloud APIs</td></tr>
 <tr><td>Memory tiering</td><td>Per layer: VRAM → pinned RAM → SSD stream, migrating live while you chat</td><td>Per expert: hot experts in VRAM, all in RAM, lookup table on SSD</td><td>None of its own (left to the backend)</td></tr>
 <tr><td>Install</td><td>One ~0.7 MB exe, no install</td><td>Installer / app, 40–70 GB model download</td><td>Docker / multi-service setup</td></tr>
 <tr><td>Models</td><td>Any supported GGUF you already have</td><td>One specific 125B model</td><td>270+ via its backends and cloud</td></tr>
 <tr><td>Hardware</td><td>Any AVX2 CPU; NVIDIA GPU optional</td><td>NVIDIA, AMD, Intel Arc, Apple Silicon; 32 GB+ memory</td><td>Whatever the chosen backend supports</td></tr></table>
 <div class="pc"><div><h4>Ithaca: pros</h4><ul><li>Tiny, instant start, fully offline, no dependencies</li><li>Uses models already on disk</li><li>Live tier migration, KV prefix cache, CUDA graphs</li><li>Built-in memory, think hard, code run</li></ul></div>
 <div><h4>Ithaca: cons</h4><ul><li>NVIDIA-only GPU path (others use CPU)</li><li>No MoE, Mamba, Gemma or vision yet</li><li>No agents, web search or speculative decoding</li><li>Single user, one request at a time</li></ul></div>
 <div><h4>Strata: pros / cons</h4><ul><li>+ Runs a 125B model on a gaming PC</li><li>+ Expert-level tiering, speculative decoding, multi-vendor GPUs</li><li>+ OpenAI/Anthropic-compatible API</li><li>− Huge downloads, needs 32 GB+ and ~80 GB disk</li><li>− First load can freeze the PC for minutes</li><li>− Tied to one model, relies on llama.cpp</li></ul></div>
 <div><h4>Odysseus: pros / cons</h4><ul><li>+ Very broad: agents, deep research, email, docs, calendar, model comparison</li><li>+ Hardware scan recommends models</li><li>+ Can mix local and cloud models</li><li>− No engine of its own; speed depends on the backend</li><li>− Heavier setup with many moving parts</li><li>− Security concerns raised by parts of the community</li></ul></div></div>`};
async function drawMem(){
  const el=$('memList');if(!el)return;const q=($('memFind').value||'').toLowerCase(),cat=$('memCat').value;
  const list=memories.filter(m=>(!cat||normCat(m.category)==cat)&&(!q||m.text.toLowerCase().includes(q)));
  const groups={};list.forEach(m=>(groups[normCat(m.category)]=groups[normCat(m.category)]||[]).push(m));
  el.innerHTML=list.length?Object.keys(CATS).filter(c=>groups[c]).map(c=>`<div class="group" style="padding-left:2px">${CATS[c]} · ${groups[c].length}</div>`+
    groups[c].sort((a,b)=>(b.pinned-a.pinned)||(b.updated-a.updated)).map(m=>`<div class="memrow" data-id="${m.id}"><span>${esc(m.text)}<small>#${m.id} · updated ${new Date(m.updated||m.created).toLocaleDateString()}${m.pinned?' · pinned':''}</small></span>
    <button class="tbtn" data-p title="${m.pinned?'Unpin':'Pin (always visible to the model)'}" style="${m.pinned?'color:var(--a1)':''}">${ico('pin')}</button><button class="tbtn" data-e title="Edit">${ico('edit')}</button><button class="tbtn" data-d title="Delete">${ico('trash')}</button></div>`).join('')).join('')
    :`<p class="sub">${memories.length?'No memories match.':'No memories yet. Tell Ithaca something about yourself, or add one here.'}</p>`;
  const save=()=>{api('/api/memories',memories);drawMem()};
  el.querySelectorAll('[data-d]').forEach(b=>b.onclick=()=>{const id=+b.parentNode.dataset.id;memories=memories.filter(m=>m.id!=id);save()});
  el.querySelectorAll('[data-p]').forEach(b=>b.onclick=()=>{const m=memories.find(x=>x.id==b.parentNode.dataset.id);m.pinned=!m.pinned;save()});
  el.querySelectorAll('[data-e]').forEach(b=>b.onclick=()=>{const row=b.parentNode,m=memories.find(x=>x.id==row.dataset.id);
    row.innerHTML=`<textarea rows="2">${esc(m.text)}</textarea><select>${Object.entries(CATS).map(([k,v])=>`<option value="${k}" ${normCat(m.category)==k?'selected':''}>${v}</option>`).join('')}</select><button class="btn pri">Save</button>`;
    row.querySelector('.btn').onclick=()=>{m.text=row.querySelector('textarea').value.trim();m.category=row.querySelector('select').value;m.updated=Date.now();save()}})}
/* ---------- model hub + system ---------- */
let sysinfo=null,bench={state:'idle',results:{}},hubTab='curated',hubData=null,dlTimer=0;
const GB=b=>(b/1073741824).toFixed(b>=10*1073741824?0:1)+' GB';
async function getSys(){if(!sysinfo)sysinfo=await api('/api/sysinfo');return sysinfo}
// How a model of `bytes` will run here: where it lives and a rough speed from measured bandwidth.
function fitFor(bytes){
  const s=sysinfo;if(!s||!bytes)return null;
  const vram=s.gpu?s.gpu.vram_mb*1048576*0.9-1.2e9:0,ram=s.ram.total_mb*1048576*0.75;
  const r=bench.results||{},vbw=(r.vram_gbps||(s.gpu?300:0))*1e9*0.55,rbw=(r.ram_gbps||25)*1e9*0.6,dbw=(r.disk_read_gbps||1.5)*1e9;
  const inV=Math.min(bytes,Math.max(0,vram)),inR=Math.min(bytes-inV,ram),inD=bytes-inV-inR;
  const t=(inV?inV/vbw:0)+(inR?inR/rbw:0)+(inD?inD/dbw:0),tps=t>0?1/t:0;
  const cls=inD>0?'slow':inR>0?'ok':'fast';
  const label=inD>0?'Streams from SSD':inR>0?(inV>0?'Split VRAM + RAM':'Runs in RAM'):'Fits in VRAM';
  return {cls,label,tps}}
function fitBadge(bytes){const f=fitFor(bytes);if(!f)return '';
  return `<span class="fit ${f.cls}" title="Rough estimate from your hardware${bench.results.ram_gbps?' and benchmark':''}">${f.label} · ~${f.tps>=10?Math.round(f.tps):f.tps.toFixed(1)} tok/s</span>`}
TABS.hub=()=>`<h2>Download models</h2><p class="sub">Get chat models straight into Ithaca's models folder. Each one shows how it would run on this PC.</p>
 <div class="seg" id="hubSeg"><button data-h="curated" class="${hubTab=='curated'?'on':''}">GPT4All picks</button><button data-h="hf" class="${hubTab=='hf'?'on':''}">Search Hugging Face</button></div>
 <div id="hubSearch" style="display:${hubTab=='hf'?'flex':'none'};gap:8px;margin:12px 0"><input id="hfQ" class="search" style="margin:0;flex:1" placeholder="e.g. qwen 7b instruct, llama 3.1 8b, mistral"><button class="btn pri" id="hfGo">Search</button></div>
 <div id="dlBox"></div><div id="hubList"><p class="sub">Loading…</p></div>`;
TABS.system=()=>`<h2>System &amp; benchmark</h2><p class="sub">What's in this PC and how fast each part is. Ithaca uses these numbers to say how models will run.</p>
 <div id="sysBox"><p class="sub">Reading hardware…</p></div>
 <div style="display:flex;align-items:center;gap:12px;margin:18px 0 10px"><button class="btn pri" id="benchGo">Run benchmark</button><span id="benchMsg" style="color:var(--dim);font-size:13px">Takes about 30 seconds. Writes a 1 GB test file to the models drive and deletes it.</span></div>
 <div id="benchBox"></div>`;
function hubRow(m){
  const st=(window._dls||[]).find(d=>d.file==m.file),have=(window._localNames||[]).includes(m.file.toLowerCase());
  const btn=have?'<span class="fit fast">Downloaded</span>':st&&(st.state=='downloading'||st.state=='queued')?`<button class="btn" data-cancel="${esc(m.file)}">Cancel</button>`
    :`<button class="btn pri" data-dl="${esc(m.file)}">Download</button>`;
  return `<div class="hubrow"><div class="hubmain"><b>${esc(m.name)}</b><span class="hubmeta">${[m.params,m.quant,m.bytes?GB(m.bytes):'',m.ram_gb?m.ram_gb+' GB RAM suggested':''].filter(Boolean).map(esc).join(' · ')} ${fitBadge(m.bytes)}</span>
    ${m.desc?`<p>${esc(m.desc.slice(0,260))}${m.desc.length>260?'…':''}</p>`:''}</div>${btn}</div>`}
async function drawHub(){
  const el=$('hubList');if(!el)return;await getSys();
  const local=await api('/api/models');window._localNames=local.map(m=>m.name.toLowerCase());
  if(hubTab=='curated'){
    if(!hubData||!hubData.models)hubData=await api('/api/hub/catalog');
    if(hubData.error){el.innerHTML=`<p class="sub">${esc(hubData.error)}</p>`;return}
    el.innerHTML=hubData.models.map(hubRow).join('');
  }else if(window._hfFiles){
    const f=window._hfFiles;
    el.innerHTML=`<button class="btn" id="hfBack">← Results</button><h4 style="margin:12px 0 6px">${esc(f.repo)}</h4>`+(f.files.length?f.files.map(x=>hubRow({name:x.path.split('/').pop(),file:x.path.split('/').pop(),url:x.url,bytes:x.bytes,quant:(x.path.match(/(IQ\d\w*|Q\d(_K)?(_[SML])?|Q\d_\d|F16|BF16|F32)/i)||[''])[0].toUpperCase()})).join(''):'<p class="sub">No .gguf files in this repository.</p>');
    $('hfBack').onclick=()=>{window._hfFiles=null;drawHub()};
  }else if(window._hfRepos){
    el.innerHTML=window._hfRepos.length?window._hfRepos.map(r=>`<div class="hubrow"><div class="hubmain"><b>${esc(r.id)}</b><span class="hubmeta">${r.downloads.toLocaleString()} downloads · ${r.likes} likes</span></div><button class="btn" data-repo="${esc(r.id)}">Files</button></div>`).join(''):'<p class="sub">Nothing found.</p>';
  }else el.innerHTML='<p class="sub">Search for a model family. Pick a Q4_K_M or Q5_K_M file for a good size/quality balance.</p>';
  bindHub()}
function hubModels(){return hubTab=='curated'?(hubData&&hubData.models||[]):(window._hfFiles?window._hfFiles.files.map(x=>({file:x.path.split('/').pop(),url:x.url,bytes:x.bytes})):[])}
function bindHub(){
  const el=$('hubList');if(!el)return;
  el.querySelectorAll('[data-dl]').forEach(b=>b.onclick=async()=>{const m=hubModels().find(x=>x.file==b.dataset.dl);if(!m)return;
    const r=await api('/api/hub/download',{file:m.file,url:m.url,bytes:m.bytes});if(r.error)toast(r.error);pollDl()});
  el.querySelectorAll('[data-cancel]').forEach(b=>b.onclick=async()=>{await api('/api/hub/cancel',{file:b.dataset.cancel});pollDl()});
  el.querySelectorAll('[data-repo]').forEach(b=>b.onclick=async()=>{el.innerHTML='<p class="sub">Listing files…</p>';
    const r=await api('/api/hub/files?repo='+encodeURIComponent(b.dataset.repo));if(r.error){toast(r.error);drawHub();return}
    window._hfFiles={repo:b.dataset.repo,files:r.files};drawHub()})}
async function pollDl(){
  clearTimeout(dlTimer);const box=$('dlBox');if(!box)return;
  const list=await api('/api/hub/downloads');const prev=window._dls||[];window._dls=list;
  box.innerHTML=list.filter(d=>d.state!='cancelled').map(d=>{const p=d.total?d.got/d.total*100:0;
    return `<div class="dlrow"><div><b>${esc(d.file)}</b><span>${d.state=='done'?'Done. Pick it in Model &amp; hardware.':d.state=='error'?'Failed: '+esc(d.err):`${GB(d.got)} of ${d.total?GB(d.total):'?'} · ${(d.speed/1048576).toFixed(1)} MB/s${d.speed&&d.total?' · '+Math.ceil((d.total-d.got)/d.speed/60)+' min left':''}${d.err?' · '+esc(d.err):''}`}</span></div><div class="dlbar"><i style="width:${d.state=='done'?100:p}%"></i></div></div>`}).join('');
  const changed=list.some(d=>{const o=prev.find(x=>x.file==d.file);return !o||o.state!=d.state});
  if(changed)drawHub();
  if(list.some(d=>d.state=='downloading'||d.state=='queued'))dlTimer=setTimeout(pollDl,1000)}
function sysCard(t,rows){return `<div class="syscard"><h4>${t}</h4>${rows.filter(r=>r&&r[1]!==''&&r[1]!=null).map(r=>`<div><span>${r[0]}</span><b>${r[1]}</b></div>`).join('')}</div>`}
async function drawSys(){
  const s=await getSys(),el=$('sysBox');if(!el)return;
  const sticks=s.ram.sticks||[];
  el.innerHTML=`<div class="sysgrid">`+
    sysCard(ico('chip')+'Processor',[['Model',esc(s.cpu.name)],['Cores / threads',s.cpu.cores+' / '+s.cpu.threads],['Base clock',s.cpu.mhz?(s.cpu.mhz/1000).toFixed(2)+' GHz':''],['Instructions',esc(s.cpu.features)]])+
    sysCard(ico('mem')+'Memory',[['Installed',(s.ram.total_mb/1024).toFixed(1)+' GB'],['Free now',(s.ram.free_mb/1024).toFixed(1)+' GB'],['Speed',s.ram.mts?s.ram.mts+' MT/s':''],['Type',sticks.length?esc(sticks[0].type||'')+' · '+sticks.length+' stick'+(sticks.length>1?'s':''):''],
      ...sticks.map(k=>[esc(k.slot||'Stick'),`${(k.mb/1024).toFixed(0)} GB ${esc(k.maker)} ${esc(k.part)}`])])+
    (s.gpu?sysCard(ico('image')+'Graphics',[['Model',esc(s.gpu.name)],['VRAM',(s.gpu.vram_mb/1024).toFixed(1)+' GB'],['Free now',(s.gpu.free_mb/1024).toFixed(1)+' GB'],['Tensor cores',s.gpu.tensor_cores?'Yes':'No'],['Driver',esc(s.gpu.driver)]])
      :sysCard(ico('image')+'Graphics',[['NVIDIA GPU','Not found. Models run on the CPU.']]))+
    sysCard(ico('code')+'Storage',[['System',esc(s.os)],...s.drives.map(d=>[d.letter+':'+(d.letter==s.models_drive?' (models)':''),`${d.ssd?'SSD':'HDD'} · ${d.free_gb} of ${d.total_gb} GB free`])])+
    `</div>`+capacityHtml(s)}
function capacityHtml(s){
  const v=s.gpu?s.gpu.vram_mb/1024:0,r=s.ram.total_mb/1024,d=(s.drives.find(x=>x.letter==s.models_drive)||{}).free_gb||0;
  const fast=Math.max(0,v*0.9-1.2),tier=fast+r*0.75;
  return `<div class="syscap"><h4>What you can run</h4>
   <div><b>${fast.toFixed(0)} GB</b><span>models fully in VRAM (fastest, e.g. ${fast>=7?'a 13B at Q4':fast>=4?'an 8B at Q4':'a 3B at Q4'})</span></div>
   <div><b>${tier.toFixed(0)} GB</b><span>with Ithaca's VRAM + RAM tiering (${tier>=40?'a 70B at Q4':tier>=18?'a 32B at Q4':tier>=8?'a 13B at Q4':'an 8B at Q4'})</span></div>
   <div><b>${(tier+d).toFixed(0)} GB</b><span>if you let it stream from the SSD too (slow, but it runs)</span></div></div>`}
function drawBench(){
  const el=$('benchBox');if(!el)return;const r=bench.results||{};
  const rows=[['CPU, one core','cpu_1t_gflops','GFLOPS'],['CPU, all cores','cpu_mt_gflops','GFLOPS'],['Memory bandwidth','ram_gbps','GB/s'],['Disk read','disk_read_gbps','GB/s'],['Disk write','disk_write_gbps','GB/s'],
    ['GPU upload (PCIe)','pcie_gbps','GB/s'],['VRAM bandwidth','vram_gbps','GB/s'],['GPU math, TF32','gpu_tf32_tflops','TFLOPS'],['GPU math, FP16','gpu_fp16_tflops','TFLOPS']];
  const max={cpu_1t_gflops:150,cpu_mt_gflops:2500,ram_gbps:100,disk_read_gbps:7,disk_write_gbps:7,pcie_gbps:26,vram_gbps:1000,gpu_tf32_tflops:80,gpu_fp16_tflops:160};
  el.innerHTML=Object.keys(r).length||bench.state=='running'?rows.filter(x=>r[x[1]]!=null||bench.state=='running').map(([n,k,u])=>`<div class="benchrow"><span>${n}</span><div class="dlbar"><i style="width:${r[k]!=null?Math.min(100,r[k]/max[k]*100):0}%"></i></div><b>${r[k]!=null?(r[k]>=100?Math.round(r[k]):r[k].toFixed(1))+' '+u:'…'}</b></div>`).join('')
    +(bench.state=='done'&&r.ram_gbps?`<p class="sub" style="margin-top:12px">${benchSummary(r)}</p>`:''):'';
  const m=$('benchMsg');if(m&&bench.state=='running')m.textContent='Testing: '+(bench.step||'…');else if(m&&bench.state=='done')m.textContent='Done. Model speed estimates now use these numbers.';
  const b=$('benchGo');if(b)b.disabled=bench.state=='running'}
function benchSummary(r){
  const ex=[4.7e9,9e9,20e9],nm=['an 8B Q4 model','a 14B Q4 model','a 32B Q4 model'];
  return 'Estimated chat speed here: '+ex.map((b,i)=>{const f=fitFor(b);return `${nm[i]} ≈ ${f.tps>=10?Math.round(f.tps):f.tps.toFixed(1)} tokens/s`}).join(', ')+
    '. The bottleneck for chat is memory bandwidth: VRAM is about '+Math.round((r.vram_gbps||0)/(r.ram_gbps||1))+'× faster than your RAM.'}
async function pollBench(){bench=await api('/api/bench');if(bench.state=='done'&&Object.keys(bench.results).length){prefs.bench=bench.results;savePrefs()}else if(bench.state=='idle'&&prefs.bench)bench={state:'done',results:prefs.bench};drawBench();if(bench.state=='running')setTimeout(pollBench,700)}
async function openHubTab(){
  $('hubSeg').querySelectorAll('button').forEach(b=>b.onclick=()=>{hubTab=b.dataset.h;openSettings('hub')});
  const go=async()=>{const q=$('hfQ').value.trim();if(!q)return;window._hfFiles=null;$('hubList').innerHTML='<p class="sub">Searching…</p>';
    const r=await api('/api/hub/search?q='+encodeURIComponent(q));if(r.error){toast(r.error);window._hfRepos=[]}else window._hfRepos=r.repos;drawHub()};
  $('hfGo').onclick=go;$('hfQ').onkeydown=e=>{if(e.key=='Enter')go()};
  if(!bench.results||!Object.keys(bench.results).length){bench=await api('/api/bench');if(bench.state=='idle'&&prefs.bench)bench={state:'done',results:prefs.bench}}
  await drawHub();pollDl()}
async function openSysTab(){
  $('benchGo').onclick=async()=>{const r=await api('/api/bench',{});if(r.error){toast(r.error);return}sysinfo=null;pollBench()};
  await pollBench();
  await drawSys()}
async function openSettings(tab){
  $('settings').classList.add('open');tab=tab||'general';
  $('tabs').querySelectorAll('button').forEach(b=>b.classList.toggle('on',b.dataset.tab==tab));
  $('pane').innerHTML=TABS[tab]();bindPane();
  if(tab=='memory'){drawMem();$('memFind').oninput=drawMem;$('memCat').onchange=drawMem;
    $('memAdd').onclick=()=>{const t=prompt('New memory');if(t&&t.trim()){memories.push({id:memNextId(),text:t.trim(),category:$('memCat').value||'other',pinned:false,created:Date.now(),updated:Date.now()});api('/api/memories',memories);drawMem()}};
    $('memClear').onclick=()=>{if(confirm('Delete all memories?')){memories=[];api('/api/memories',memories);drawMem()}};
    $('memReview').onclick=async()=>{
      if(!status.loaded){toast('Load a model first');return}if(!memories.length){toast('No memories to review');return}
      const b=$('memReview');b.disabled=true;b.innerHTML='<span class="dots"><span></span><span></span><span></span></span> Reviewing…';
      const r=await stream([{role:'system',content:`You are the curator of a long-term memory list about one user. Review every entry and improve the list:
- Merge duplicates and near-duplicates: keep one (edit it to combine the details) and delete the others.
- Rewrite vague or messy entries into one short, specific, third-person sentence ("User ...").
- Fix the category when it is wrong. Categories: about, preferences, projects, knowledge, other.
- Delete entries that are useless, contradictory, temporary, or contain secrets.
Reply ONLY with tags, one per line:
<memory_edit id="N" category="...">full new text</memory_edit>
<memory_delete id="N"/>
<memory_save category="...">text</memory_save>
If the list is already good, reply with nothing.`},
        {role:'user',content:'Memories:\n'+memories.map(memLine).join('\n')}],{max_tokens:1500,temperature:0.2});
      const ops=applyMemoryOps(parseTools(r.text));b.disabled=false;b.textContent='Review & clean up with AI';toast(ops.length?`Review done: ${ops.length} change(s)`:'Memories already look good');drawMem()}}
  if(tab=='hub')openHubTab();if(tab=='system')openSysTab();
  if(tab=='model'){
    const list=await api('/api/models');const s=$('mSel');
    s.innerHTML=list.map(m=>`<option value="${esc(m.path)}" ${m.path==status.model?'selected':''}>${esc(m.name)} (${(m.mb/1024).toFixed(1)} GB)</option>`).join('')||'<option value="">No .gguf found. Add a folder below</option>';
    $('mAdd').onclick=async()=>{await api('/api/adddir',{dir:$('mDir').value});openSettings('model')};
    $('mLoad').onclick=async()=>{const b=$('mLoad');b.disabled=true;$('mMsg').textContent='Loading…';
      const r=await api('/api/load',{path:s.value,threads:+$('mThreads').value,ctx:+$('mCtx').value,vram_mb:+$('mVram').value,ram_mb:+$('mRam').value});
      b.disabled=false;$('mMsg').textContent=r.error?'Failed: '+r.error:`Loaded in ${r.load_ms} ms. Layers move to VRAM/RAM in the background.`;refreshStatus().then(updateCtx)}}}
$('tabs').querySelectorAll('button').forEach(b=>b.onclick=()=>openSettings(b.dataset.tab));
$('openSettings').onclick=()=>openSettings();
$('quitApp').onclick=async()=>{if(busy&&!confirm('A reply is still being written. Quit anyway?'))return;if(!busy&&!confirm('Quit Ithaca? This stops it and frees all of its memory.'))return;
  try{await fetch('/api/quit',{method:'POST'})}catch(e){}
  document.body.innerHTML='<div style="display:grid;place-items:center;height:100vh;font-family:inherit;color:var(--dim);text-align:center"><div><h2 style="color:var(--text);margin:0 0 6px">Ithaca has stopped</h2>All memory is freed. You can close this tab, and start Ithaca again any time.</div></div>'};$('openHub').onclick=()=>openSettings('hub');$('modelPick').onclick=()=>openSettings('model');
$('closeSettings').onclick=()=>{$('settings').classList.remove('open');render()};
$('settings').onclick=e=>{if(e.target.id=='settings'){$('settings').classList.remove('open');render()}};

/* ---------- studio: image / video / audio ---------- */
let view='chat',mediaModels=null;
let vsel={amode:'Speech'};
function setView(v){const prev=view;view=v;
  if(prev!=v)setTimeout(()=>{const t=v=='chat'?$('log'):$('studio');if(t){t.classList.remove('viewin');void t.offsetWidth;t.classList.add('viewin')}});$('nav').querySelectorAll('button').forEach(b=>b.classList.toggle('on',b.dataset.view==v));
  const chatv=v=='chat';$('log').style.display=chatv?'':'none';document.querySelector('.composer').style.display=chatv?'':'none';
  document.querySelector('.ctxwrap').style.visibility=chatv?'':'hidden';$('studio').style.display=chatv?'none':'block';
  if(!chatv)drawStudio()}
function modelSelect(kind){const list=(mediaModels||[]).filter(m=>m.kind==kind),cur=prefs['media_'+kind]||'';
  return `<select id="mediaSel">${'<option value="">None selected</option>'+list.map(m=>`<option value="${esc(m.path)}" ${m.path==cur?'selected':''}>${esc(m.name)} (${(m.mb/1024).toFixed(1)} GB)</option>`).join('')}</select>
   <small>${list.length?`${list.length} ${kind} model(s) found on this PC.`:`No ${kind} models found yet. Put them in the <b>models</b> folder next to Ithaca.exe, or add their folder below.`}</small>
   <div style="display:flex;gap:8px;margin-top:8px"><input id="mdir" placeholder="Add a folder that contains ${kind} models"><button class="btn" id="mdirAdd">Add folder</button></div>`}
function opts(key,vals){return `<div class="opts" data-key="${key}">`+vals.map(v=>`<button class="opt${vsel[key]==v?' on':''}">${v}</button>`).join('')+'</div>'}
async function polish(kind,text,out){
  if(!status.loaded){toast('Load a chat model first to polish prompts');return}
  out.textContent='Polishing…';
  const guide={image:'Rewrite it as one vivid, specific image-generation prompt: subject, composition, lighting, style, lens/medium, colour palette and quality terms. Under 80 words.',
    video:'Rewrite it as one strong text-to-video prompt. Describe one continuous shot: the subject and its appearance, a single clear action that fits the duration, the setting, camera movement, lighting and style. Use plain present-tense sentences, no lists, under 100 words. Avoid anything that needs text rendering, many people, or scene cuts, since video models handle those poorly.',
    audio:'Rewrite it as a clear prompt for an audio model: what is heard, mood, tempo/pace, instruments or voice character, and length. Under 60 words.'}[kind];
  const r=await stream([{role:'system',content:'You write prompts for generative media models. '+guide+' Reply with the prompt only.'},{role:'user',content:text}],{max_tokens:300,temperature:0.6});
  out.textContent=stripTags(splitThink(r.text).body).trim()}
async function drawStudio(){
  if(!mediaModels)mediaModels=await api('/api/media/models');
  const kind=view,st=$('studio');
  const banner=kind=='image'
    ?`<div class="banner" style="border-left-color:var(--ok)"><b>Built-in engine:</b> runs single-file <b>SD 1.5, SD 2.x and SDXL</b> checkpoints (including fine-tunes like Pony, Illustrious, Juggernaut). Layout is read from the file, so any checkpoint of these families works. Weights that don't fit in free VRAM are streamed from RAM. You can also just ask for an image in a normal chat. Flux and SD3 support is coming.</div>`
    :kind=='audio'?`<div class="banner" style="border-left-color:var(--ok)"><b>Built-in engine:</b> speech with <b>F5-TTS</b> voice cloning and the Vocos vocoder. Pick a default voice and the chat can talk too ("say this out loud…"). Music, sound effects, Kokoro/Chatterbox voices and speech-to-text are next.</div>`
    :kind=='video'?`<div class="banner" style="border-left-color:var(--ok)"><b>Built-in engine:</b> runs <b>Wan 2.2 TI2V-5B</b> text-to-video (with the umT5-XXL text encoder and Wan 2.2 VAE, found automatically next to the model). Weights that don't fit in VRAM stream from RAM. You can also ask for a video in a normal chat. Image-to-video and the 14B GGUF models are coming.</div>`
    :`<div class="banner"><b>Engine status:</b> Ithaca can already find and select ${kind} models and prepare prompts, and the chat model knows how to request ${kind}s. The ${kind} engine itself is still being built, so Generate reports that for now.</div>`;
  let body='';
  const adv=inner=>`<details class="card adv"><summary>More options</summary><div class="in">${inner}</div></details>`;
  if(kind=='image')body=`<div class="card hero"><textarea id="p_text" rows="3" placeholder="Describe the picture. For example: a misty mountain lake at sunrise, as a background for my iPhone"></textarea>
   <div class="sizerow"><label>Size</label><input id="p_w" type="number" min="256" max="2048" step="8" value="${prefs.img_width||1024}"><span class="x">×</span><input id="p_h" type="number" min="256" max="2048" step="8" value="${prefs.img_height||1024}">
   <button class="btn" id="p_swap" title="Swap width and height">⇄</button><label class="chk"><input type="checkbox" id="p_auto" ${prefs.img_auto!==false?'checked':''}>Pick from the description</label>
   <label class="chk"><input type="checkbox" id="p_fit" ${prefs.img_fit!==false?'checked':''}>Best quality for this model</label><span class="hint" id="p_hint"></span></div>
   <div class="row"><button class="btn" id="polish">${ico('wand')} Improve prompt</button><button class="btn pri" id="gen">Generate image</button></div>
   <div class="preview" id="pv" style="margin-top:12px;display:none"></div><div class="result" id="res"></div></div>
   ${adv(`<div class="field"><label>Model</label>${modelSelect('image')}</div>
   <div class="field"><label>Avoid (negative prompt)</label><input id="p_neg" value="${esc(prefs.img_negative||'blurry, low quality, deformed, watermark, text')}"></div>
   <div class="grid2"><div class="field"><label>Steps · Guidance (CFG) · Seed</label><div style="display:flex;gap:8px"><input id="p_steps" type="number" min="1" max="150" value="${prefs.img_steps||25}"><input id="p_cfg" type="number" step="0.5" value="${prefs.img_cfg||6}"><input id="p_seed" placeholder="random"></div></div>
   <div class="field"><label>Sampler · Schedule</label><div style="display:flex;gap:8px"><select id="p_sampler">${[['dpmpp_2m','DPM++ 2M'],['euler_a','Euler a'],['euler','Euler']].map(([v,t])=>`<option value="${v}" ${v==(prefs.img_sampler||'dpmpp_2m')?'selected':''}>${t}</option>`).join('')}</select><select id="p_sched">${['karras','normal'].map(v=>`<option ${v==(prefs.img_scheduler||'karras')?'selected':''}>${v}</option>`).join('')}</select></div></div></div>
   <div class="field"><label>VAE (optional)</label><input id="p_vae" placeholder="Leave empty to use the checkpoint's own VAE" value="${esc(prefs.media_image_vae||'')}"></div>`)}
   <div class="card"><h3>Gallery</h3><div class="gallery" id="gal"></div></div>`;
  if(kind=='video')body=`<div class="card hero"><textarea id="v_text" rows="3" placeholder="Describe one shot: a red fox trotting through fresh snow at dawn, slow push-in, cinematic"></textarea>
   <div class="sizerow"><label>Length</label><input id="v_len" type="range" min="1" max="5" step="1" value="${prefs.vid_seconds||2}"><span class="hint" id="v_lenl"></span>
   <label style="margin-left:10px">Size</label><input id="v_w" type="number" min="256" max="1280" step="32" value="${prefs.vid_width||640}"><span class="x">×</span><input id="v_h" type="number" min="256" max="1280" step="32" value="${prefs.vid_height||352}"><button class="btn" id="v_swap" title="Swap width and height">⇄</button></div>
   <div class="row"><button class="btn" id="polish">${ico('wand')} Improve prompt</button><button class="btn pri" id="gen">Generate video</button></div>
   <div class="preview" id="pv" style="margin-top:12px;display:none"></div><div class="result" id="res"></div>
   <small style="color:var(--faint)">One subject, one motion, one camera move works best. 640×352 for 2 seconds takes about 3 minutes on an RTX 3060.</small></div>
   ${adv(`<div class="field"><label>Model</label>${modelSelect('video')}</div>
   <div class="grid2"><div class="field"><label>Steps</label><input id="v_steps" type="number" min="4" max="60" value="${prefs.vid_steps||20}"></div><div class="field"><label>Guidance (CFG)</label><input id="v_cfg" type="number" step="0.5" value="${prefs.vid_cfg||5}"></div></div>`)}
   <div class="card"><h3>Gallery</h3><div class="gallery" id="gal"></div></div>`;
  if(kind=='audio')body=`<div class="card"><h3>Speech model</h3><div class="field">${modelSelect('audio')}</div>
   <small style="color:var(--faint)">Supported: F5-TTS (the .safetensors with vocab.txt next to it, and a vocos folder beside it).</small></div>
   <div class="card"><h3>Voices</h3><p class="sub" style="margin-bottom:10px">Ithaca speaks in the voice of a short reference clip (5–12 seconds of clean speech, WAV). Type exactly what is said in it.</p>
   <div id="voices"></div>
   <div class="grid2" style="margin-top:12px"><div class="field"><label>Name</label><input id="vn" placeholder="e.g. Narrator"></div><div class="field"><label>Clip (.wav)</label><input id="vf" type="file" accept=".wav,audio/wav"></div></div>
   <div class="field"><label>Exact words spoken in the clip</label><input id="vt" placeholder="Type the transcript of the clip"></div>
   <button class="btn" id="vadd">+ Add voice</button></div>
   <div class="card"><h3>What to make</h3>${opts('amode',['Speech','Music','Sound effect','Transcribe'])}
   <div class="field" style="margin-top:12px"><textarea id="p_text" rows="3" placeholder="Text to speak"></textarea></div>
   <div class="grid2"><div class="field"><label>Speed</label><input id="a_speed" type="number" step="0.1" min="0.5" max="2" value="${prefs.tts_speed||1}"></div></div>
   <div style="display:flex;gap:8px"><button class="btn" id="polish">${ico('wand')} Improve with AI</button><button class="btn pri" id="gen">Generate audio</button></div>
   <div class="preview" id="pv" style="margin-top:12px;display:none"></div><div class="result" id="res"></div>
   <small style="color:var(--faint)">Speech is built in. Music, sound effects and transcription need engines that aren't built yet.</small></div>
   <div class="card"><h3>Gallery</h3><div class="gallery" id="gal"></div></div>`;
  st.innerHTML=`<div class="col studio"><h1>${ico(kind)}${kind[0].toUpperCase()+kind.slice(1)} studio</h1>${banner}${body}</div>`;
  if(!vsel.amode)vsel.amode='Speech';
  st.querySelectorAll('.opts').forEach(g=>g.querySelectorAll('.opt').forEach(b=>b.onclick=()=>{vsel[g.dataset.key]=b.textContent;g.querySelectorAll('.opt').forEach(x=>x.classList.toggle('on',x==b))}));
  $('mediaSel').onchange=e=>{prefs['media_'+kind]=e.target.value;savePrefs()};
  $('mdirAdd').onclick=async()=>{const d=$('mdir').value.trim();if(!d)return;mediaModels=await api('/api/media/adddir',{dir:d});
    prefs=Object.assign(prefs,await api('/api/prefs'));toast('Folder added');drawStudio()};
  if(kind=='audio'){drawVoices();drawGallery();$('a_speed').onchange=()=>{prefs.tts_speed=+$('a_speed').value;savePrefs()};
    $('vadd').onclick=async()=>{const f=$('vf').files[0],name=$('vn').value.trim(),text=$('vt').value.trim();
      if(!f||!name||!text){toast('Add a name, a WAV clip and its transcript');return}
      const wav=await new Promise(r=>{const fr=new FileReader();fr.onload=()=>r(fr.result);fr.readAsDataURL(f)});
      const res=await api('/api/voices/add',{name,text,wav});if(res.error){toast(res.error);return}
      if(!prefs.voice){prefs.voice=name;savePrefs()}toast('Voice added');drawVoices()}}
  const promptEl=$(kind=='video'?'v_text':'p_text');
  $('polish').onclick=()=>{const pv=$('pv');pv.style.display='block';polish(kind,promptEl.value||'',pv)};
  if(kind=='image'){drawGallery();
    // the size follows the description ("...for my iPhone") until it is typed in by hand; the hint shows what will be rendered
    const hint=()=>{const w=+$('p_w').value||1024,h=+$('p_h').value||1024;let rw=w,rh=h;
      if($('p_fit').checked){const a=w/h,res=1024;rw=Math.round(Math.sqrt(res*res*a)/64)*64;rh=Math.round(rw/a/64)*64}
      $('p_hint').textContent=$('p_fit').checked&&(rw!=w||rh!=h)?`renders at about ${rw} × ${rh}, same shape`:''};
    const auto=()=>{if(!$('p_auto').checked)return;const z=pickSize({},promptEl.value);const k=1024/Math.sqrt(z.width*z.height);
      $('p_w').value=Math.round(z.width*k/8)*8;$('p_h').value=Math.round(z.height*k/8)*8;hint()};
    const save=()=>{prefs.img_negative=$('p_neg').value;prefs.img_steps=+$('p_steps').value;prefs.img_cfg=+$('p_cfg').value;prefs.img_sampler=$('p_sampler').value;
      prefs.img_scheduler=$('p_sched').value;prefs.media_image_vae=$('p_vae').value.trim();prefs.img_width=+$('p_w').value;prefs.img_height=+$('p_h').value;
      prefs.img_auto=$('p_auto').checked;prefs.img_fit=$('p_fit').checked;savePrefs();hint()};
    ['p_neg','p_steps','p_cfg','p_sampler','p_sched','p_vae','p_fit','p_auto'].forEach(id=>$(id).onchange=save);
    ['p_w','p_h'].forEach(id=>$(id).oninput=()=>{$('p_auto').checked=false;save()});
    $('p_swap').onclick=()=>{const w=$('p_w').value;$('p_w').value=$('p_h').value;$('p_h').value=w;$('p_auto').checked=false;save()};
    promptEl.oninput=auto;hint()}
  if(kind=='video'){drawGallery();
    const len=()=>{$('v_lenl').textContent=$('v_len').value+' s'};len();
    const save=()=>{prefs.vid_seconds=+$('v_len').value;prefs.vid_width=+$('v_w').value;prefs.vid_height=+$('v_h').value;prefs.vid_steps=+$('v_steps').value;prefs.vid_cfg=+$('v_cfg').value;savePrefs();len()};
    ['v_len','v_w','v_h','v_steps','v_cfg'].forEach(id=>$(id).onchange=save);$('v_len').oninput=len;
    $('v_swap').onclick=()=>{const w=$('v_w').value;$('v_w').value=$('v_h').value;$('v_h').value=w;save()}}
  $('gen').onclick=async()=>{const prompt=($('pv').style.display!='none'&&$('pv').textContent)||promptEl.value;
    if(!prompt.trim()){toast('Write a prompt first');return}
    const extra={};
    if(kind=='image'){Object.assign(extra,{width:+$('p_w').value,height:+$('p_h').value,fit:$('p_fit').checked,steps:+$('p_steps').value,cfg:+$('p_cfg').value,
      sampler:$('p_sampler').value,scheduler:$('p_sched').value,negative:$('p_neg').value,vae:$('p_vae').value.trim()});if($('p_seed').value)extra.seed=+$('p_seed').value}
    if(kind=='audio'){Object.assign(extra,{speed:+$('a_speed').value})}
    if(kind=='video'){const r32=v=>Math.max(256,Math.round(v/32)*32);Object.assign(extra,{width:r32(+$('v_w').value),height:r32(+$('v_h').value),steps:+$('v_steps').value,cfg:+$('v_cfg').value,seconds:+$('v_len').value})}
    const b=$('gen');b.disabled=true;const stop=watchProgress($('res'));
    const r=await api('/api/media/generate',Object.assign({kind,prompt,mode:vsel.amode},extra));
    stop();b.disabled=false;
    const media=r.url?(kind=='video'?`<video class="genimg" controls autoplay loop src="${esc(r.url)}"></video>`:kind=='audio'?`<audio controls src="${esc(r.url)}"></audio>`:`<div class="gwrap"><img class="genimg" src="${esc(r.url)}" onclick="viewImg(this.src)"></div>`):'';
    $('res').innerHTML=r.url?`${media}<div class="meta" style="margin:0">${r.width?r.width+' × '+r.height+' · ':''}seed ${r.seed} · ${r.seconds} s</div>`:`<span class="err">${esc(r.error)}</span>`;
    if(r.url)drawGallery()}}
async function drawVoices(){const el=$('voices');if(!el)return;const list=await api('/api/voices');
  el.innerHTML=list.length?list.map(v=>`<div class="memrow"><span><b>${esc(v.name)}</b>${prefs.voice==v.name?' <small style="display:inline;color:var(--ok)">(default)</small>':''}<small>"${esc(v.text.slice(0,90))}"</small></span>
    <audio controls src="${esc(v.url)}" style="height:32px;width:180px"></audio><button class="btn" data-def="${esc(v.name)}">${prefs.voice==v.name?'Default':'Use'}</button><button class="tbtn" data-del="${esc(v.name)}" title="Delete">${ico('trash')}</button></div>`).join('')
    :'<p class="sub">No voices yet. Add one below.</p>';
  el.querySelectorAll('[data-def]').forEach(b=>b.onclick=()=>{prefs.voice=b.dataset.def;savePrefs();drawVoices()});
  el.querySelectorAll('[data-del]').forEach(b=>b.onclick=async()=>{if(!confirm('Delete this voice?'))return;await api('/api/voices/delete',{name:b.dataset.del});if(prefs.voice==b.dataset.del){prefs.voice='';savePrefs()}drawVoices()})}
async function drawGallery(){const g=$('gal');if(!g)return;const list=(await api('/api/media/gallery')).filter(x=>(x.kind||'image')==view);
  const st='width:100%;aspect-ratio:1;object-fit:cover;border-radius:10px';
  g.innerHTML=list.length?list.map(x=>x.kind=='video'?`<video src="${esc(x.url)}" title="${esc(x.prompt)}" muted loop onmouseover="this.play()" onmouseout="this.pause()" controls style="${st}"></video>`
    :x.kind=='audio'?`<div title="${esc(x.prompt)}"><audio controls src="${esc(x.url)}" style="width:100%"></audio><small>${esc((x.prompt||'').slice(0,60))}</small></div>`
    :`<a href="${esc(x.url)}" target="_blank" title="${esc(x.prompt)}"><img src="${esc(x.url)}" loading="lazy" style="${st}"></a>`).join(''):`<div class="ph">Nothing yet</div>`}
// Shows a live progress bar in `el` while a media job runs; returns a stop function.
// Live render card: the picture sharpens step by step (latent previews), with progress and cancel.
function watchProgress(el){let on=true,seen=-1;
  el.innerHTML=`<div class="rcard"><div class="rframe"><div class="rshimmer" style="${phase(1600)}"></div><img class="rimg" alt=""><div class="rbadge"><span class="rdot"></span><span class="mp-l">Starting</span></div></div>
    <div class="rfoot"><div class="rbar"><i class="mp-b"></i></div><span class="rpct mp-p">0%</span><button class="btn mp-c">Cancel</button></div></div>`;
  el.querySelector('.mp-c').onclick=()=>fetch('/api/media/cancel',{method:'POST'});
  const fr=el.querySelector('.rframe');let img=el.querySelector('.rimg');
  // Each preview loads off-screen and fades in over the previous one, so the frame never flashes back to empty.
  const show=src=>{const n=new Image();n.className='rimg';n.alt='';
    n.onload=()=>{if(!on)return;fr.insertBefore(n,img.nextSibling);fr.classList.add('has');
      requestAnimationFrame(()=>{n.classList.add('in');const old=img;img=n;setTimeout(()=>{if(old!==img)old.remove()},600)})};
    n.src=src};
  (async()=>{while(on){try{const p=await api('/api/media/progress');if(!on)break;
    const l=el.querySelector('.mp-l'),b=el.querySelector('.mp-b'),pc=el.querySelector('.mp-p');
    if(p.busy){const f=p.total?Math.min(1,(p.step+(p.phase=='Decoding'?1:0))/p.total):0;
      l.textContent=p.total?`${p.phase} · ${p.step}/${p.total}`:p.phase;b.style.width=Math.max(3,f*100)+'%';pc.textContent=Math.round(f*100)+'%';
      if(p.preview&&p.preview!=seen){seen=p.preview;show('/api/media/preview?'+seen)}}}catch(e){}
    await new Promise(r=>setTimeout(r,400))}})();
  return()=>{on=false}}
$('nav').querySelectorAll('button').forEach(b=>b.onclick=()=>setView(b.dataset.view));

/* ---------- composer ---------- */
function autosize(){const q=$('q');q.style.height='auto';q.style.height=Math.min(240,q.scrollHeight)+'px';if(!busy)$('send').disabled=!canSend()}
$('attBtn').onclick=()=>$('fileIn').click();
$('fileIn').onchange=e=>{addFiles([...e.target.files]);e.target.value=''};
$('q').addEventListener('paste',e=>{const cd=e.clipboardData||{},fs=[...cd.files||[]];if(fs.length){e.preventDefault();addFiles(fs);return}
  if(window._clipImg&&!(cd.getData&&cd.getData('text'))){e.preventDefault();addFiles([window._clipImg])}});
document.addEventListener('dragover',e=>{if(view!='chat'||!e.dataTransfer.types.includes('Files'))return;e.preventDefault();$('box').classList.add('drop')});
document.addEventListener('dragleave',e=>{if(!e.relatedTarget)$('box').classList.remove('drop')});
document.addEventListener('drop',e=>{if(view!='chat'||!e.dataTransfer.files.length)return;e.preventDefault();$('box').classList.remove('drop');addFiles([...e.dataTransfer.files])});
$('q').oninput=autosize;
$('q').onkeydown=e=>{if(e.key=='Enter'&&!e.shiftKey){e.preventDefault();send()}};
$('send').onclick=send;
$('hardBtn').onclick=()=>{hard=!hard;$('hardBtn').classList.toggle('on',hard)};
$('newchat').onclick=newChat;$('search').oninput=drawHistory;
$('toggleSide').onclick=()=>$('side').classList.toggle('hidden');
document.addEventListener('keydown',e=>{if(e.key=='Escape'){if($('settings').classList.contains('open')){$('settings').classList.remove('open');render()}else if(busy)stopGen()}});

(async()=>{
  try{prefs=Object.assign(prefs,await api('/api/prefs'));memories=await api('/api/memories')}catch(e){}
  savePrefs();await refreshStatus();newChat();loadHistory();
  {const h=location.hash.match(/chat=([\w-]+)/);if(h)openChat(h[1])}
  setInterval(refreshStatus,2000);
})();
</script></body></html>)HTML";
