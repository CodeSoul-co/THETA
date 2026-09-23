import { StringDecoder } from 'node:string_decoder';

/** Keep both setup context and the final computed result/traceback. Never present
 * a clipped stream as complete, and do not break UTF-8 across pipe chunks. */
export class OutputCapture {
  private readonly decoder=new StringDecoder('utf8');
  private first='';private tail='';private units=0;private bytes=0;private ended=false;
  constructor(private readonly limit:number) {if(limit<512)throw new Error('Output limit too small');}
  push(chunk:Buffer) {if(this.ended)throw new Error('Output already closed');this.bytes+=chunk.length;this.append(this.decoder.write(chunk));}
  private append(text:string) {this.units+=text.length;this.first=(this.first+text).slice(0,this.limit);this.tail=(this.tail+text).slice(-(this.limit-Math.floor(this.limit/3)-128));}
  finish() {
    if(!this.ended){this.append(this.decoder.end());this.ended=true;}
    const truncated=this.units>this.limit,head=this.first.slice(0,Math.floor(this.limit/3));
    return {text:truncated?head+`\n[OUTPUT TRUNCATED: ${this.units-head.length-this.tail.length} UTF-16 units omitted; beginning and end retained]\n`+this.tail:this.first,
      bytes:this.bytes,truncated};
  }
}
