import React from "react";

// reactstrap components
// import {
// } from "reactstrap";

// core components
import IndexHeader from "components/Headers/IndexHeader.js";

// sections for this page

import { WebChatContainer, setEnableDebug } from '@ibm-watson/assistant-web-chat-react';

const webChatOptions = {
  integrationID: "66299481-ad65-4579-b1e3-09be3ccab391", 
  region: "eu-de", 
  serviceInstanceID: "08d6a0f9-b897-4316-88e9-8be83d320bb3",
};

function Index() {
  React.useEffect(() => {
    document.body.classList.add("index-page");
    document.body.classList.add("sidebar-collapse");
    document.documentElement.classList.remove("nav-open");
    window.scrollTo(0, 0);
    document.body.scrollTop = 0;
    return function cleanup() {
      document.body.classList.remove("index-page");
      document.body.classList.remove("sidebar-collapse");
    };
  });
  return (
    <>
      <WebChatContainer config={webChatOptions} />
      <div className="wrapper">
        <IndexHeader />
        <div className="main">
        </div>
      </div>
    </>
  );
}

export default Index;
